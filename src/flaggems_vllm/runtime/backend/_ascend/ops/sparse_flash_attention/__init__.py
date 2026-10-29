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


"""Shared-KV MLA with the vLLM-Ascend SFA calling convention.

Dispatch uses shape/layout metadata, not prefill/decode labels. T is the total
query row count (query.shape[0] for TND, B * S_capacity for BSND, including
padding); H is the local query head count, K the sparse entry count, and P the
KV page size. Noncontiguous Q and metadata are copied by _sfa_copy_strided;
strided KV is gathered directly or addressed through compatible row remapping.

For valid inputs, the following fast paths require paged KV and
sparse_block_size=1. Paths 1-3 also require contiguous key and key_rope.
The first matching path is used:

1. H=4, 2<T<=16, P=128, K=2048:
   _sfa_pipeline fuses gather/QK, softmax/PV and output updates in one kernel.
   Actual H4 inputs are zero-padded to an H16 Cube tile; only H4 is written back.
   Each query uses one task without KV splitting. T<=2 uses path 2 instead.
2. T<=16, H<=64, 256<=K<=4096, excluding path 1:
   _sfa_gather_sparse_kv -> _sfa_tiled_attention -> _sfa_split_combine.
   Head groups and KV splits supply parallel work, with 16-head/256-KV tiles.
   Split counts use the device's Cube count; padded sparse tails are masked.
3. H=64, T>16, P=128, K=2048, main KV cache<=2 GiB:
   _sfa_prepare_indices -> _sfa_pipeline with cross-query scheduling.
   Each core pipelines successive queries through gather/QK, softmax/PV and
   output updates, using two state buffers to keep their normalization separate.
4. H=128, P=128, K=2048 (the existing fast path):
   - T=1, or strided KV that cannot use the pipeline after row remapping:
     _sfa_gather_sparse_kv -> _sfa_gathered_attention. T=1 uses four KV splits
     and then _sfa_combine; other queries use one split without a combine.
   - Otherwise, optional _sfa_prepare_indices (including strided row remapping) for
     T>=16 and supported row bounds -> ordinary _sfa_pipeline. T=4 uses four
     splits and _sfa_combine; T=8 uses two splits and _sfa_combine_two. Other T
     values use one split. Compatible strided row remapping requires T>=16.
5. Other supported inputs use the general schedule, including nonpaged,
   block-sparse and shapes outside the fast-path gates:
   - With at least 256 expanded KV rows and either strided KV, H>16, block
     sparsity, or enough query/head tasks to occupy half the Cube cores:
     gather -> _sfa_tiled_attention (16/32 heads, 256 KV rows). Multiple splits
     use _sfa_split_combine; a single split writes output/stats directly.
     This route requires total scratch <=64 MiB.
   - Otherwise, _sfa_general loads KV directly with 16-head tiles. KV splits
     fill idle cores when useful, followed by _sfa_split_combine; one split
     writes output/stats directly. KV tiles remain 32/8/4 rows for contiguous,
     strided and wide-offset strided caches respectively.
   Both schedules balance the number of launch waves and KV iterations using
   the device core count. Unsupported contracts raise.

Core counts are queried on the input device; execution requires a 1:2
Cube/Vector ratio. Grids and small/general splits adapt to those counts, while
tiles, dispatch thresholds and H128 split choices remain fixed. Performance
has been validated on 910B3; other 910B variants require separate validation.
No NVIDIA autotune is applied.
"""

import math
from dataclasses import dataclass, replace
from typing import NamedTuple

import torch
import triton

from . import common as cfg
from .attention import (
    _sfa_combine,
    _sfa_combine_two,
    _sfa_gathered_attention,
    _sfa_general,
    _sfa_split_combine,
    _sfa_tiled_attention,
)
from .common import (
    _get_core_num,
    _sfa_gather_sparse_kv,
    _sfa_make_contiguous,
    _sfa_prepare_indices,
)
from .pipeline import _sfa_pipeline


@dataclass(frozen=True)
class _SFAContext:
    """Validated inputs and logical shape metadata; no device data is read back."""

    query: object
    key: object
    query_rope: object
    key_rope: object
    sparse_indices: object
    block_table: object
    actual_seq_lengths_query: object
    actual_seq_lengths_kv: object
    scale_value: float
    layout_query: str
    layout_kv: str
    sparse_block_size: int
    sparse_mode: int
    return_softmax_lse: bool
    num_requests: int
    num_query_tokens: int
    num_query_heads: int
    query_seq_capacity: int
    kv_seq_capacity: int
    num_sparse_blocks: int
    paged: bool
    page_size: int
    pages_per_request: int
    all_inputs_contiguous: bool

    @property
    def query_lengths(self):
        return (
            self.actual_seq_lengths_query
            if self.actual_seq_lengths_query is not None
            else self.sparse_indices
        )

    @property
    def kv_lengths(self):
        return (
            self.actual_seq_lengths_kv
            if self.actual_seq_lengths_kv is not None
            else self.sparse_indices
        )


@dataclass(frozen=True)
class _SFASchedule:
    """Host-only routing and compile-time launch choices for one invocation."""

    path: str
    num_vector_cores: int
    kv_access_spec: object = None
    key_rope_access_spec: object = None
    generic_access_requires_int64_offsets: bool = False
    cache_storage_row_span: int = 0
    shared_kv_row_layout: object = None
    num_programs: int = 0
    num_kv_splits: int = 1
    pipeline_heads: int = 0
    workspace_padding: int = 0
    use_small_head_pipeline: bool = False
    use_cross_query_pipeline: bool = False
    use_prepared_physical_indices: bool = False
    padded_sparse: int = 0
    general_kv_rows: int = 0
    use_general_gather: bool = False
    gathered_head_block: int = cfg.SMALL_QUERY_HEAD_BLOCK


class _PipelineWorkspace(NamedTuple):
    kv_workspace: torch.Tensor
    rope_workspace: torch.Tensor
    score_workspace: torch.Tensor
    unnormalized_weights_workspace: torch.Tensor
    block_weighted_value_sum_workspace: torch.Tensor
    softmax_state_workspace: torch.Tensor
    output_accumulator_workspace: torch.Tensor


def _validate_inputs(
    query,
    key,
    value,
    sparse_indices,
    scale_value,
    *,
    block_table=None,
    actual_seq_lengths_query=None,
    actual_seq_lengths_kv=None,
    query_rope=None,
    key_rope=None,
    sparse_block_size=1,
    layout_query="BSND",
    layout_kv="BSND",
    sparse_mode=cfg.SPARSE_MODE_CAUSAL,
    pre_tokens=cfg.INT64_MAX,
    next_tokens=cfg.INT64_MAX,
    attention_mode=cfg.MLA_ABSORB_ATTENTION_MODE,
    return_softmax_lse=False,
):
    """Validate the public contract and describe its logical dimensions."""
    required_tensors = (query, key, value, sparse_indices)
    if not all(isinstance(tensor, torch.Tensor) for tensor in required_tensors):
        raise TypeError("query, key, value and sparse_indices must be tensors")
    if layout_query not in ("BSND", "TND") or layout_kv not in (
        "BSND",
        "TND",
        "PA_BSND",
    ):
        raise NotImplementedError("unsupported query/cache layout")
    if layout_kv != "PA_BSND" and layout_kv != layout_query:
        raise ValueError("nonpaged query and cache layouts must match")
    if attention_mode != cfg.MLA_ABSORB_ATTENTION_MODE or sparse_mode not in (
        cfg.SPARSE_MODE_ALL,
        cfg.SPARSE_MODE_CAUSAL,
    ):
        raise NotImplementedError("only MLA absorb and sparse_mode 0/3 are supported")
    if pre_tokens != cfg.INT64_MAX or next_tokens != cfg.INT64_MAX:
        raise NotImplementedError("pre_tokens and next_tokens require their defaults")
    if (
        not isinstance(sparse_block_size, int)
        or sparse_block_size not in cfg.SUPPORTED_SPARSE_BLOCK_SIZES
    ):
        raise NotImplementedError("sparse_block_size must be a power of two in [1,128]")
    if not isinstance(return_softmax_lse, bool):
        raise TypeError("return_softmax_lse must be a bool")
    if query_rope is None or key_rope is None:
        raise NotImplementedError("this MLA interface requires separate query/key RoPE")
    optional_tensors = (
        block_table,
        actual_seq_lengths_query,
        actual_seq_lengths_kv,
        query_rope,
        key_rope,
    )
    input_tensors = required_tensors + tuple(
        tensor for tensor in optional_tensors if tensor is not None
    )
    if not all(isinstance(tensor, torch.Tensor) for tensor in input_tensors):
        raise TypeError("optional tensor arguments must be tensors or None")
    if query.device.type != "npu":
        raise NotImplementedError("sparse_flash_attention requires an Ascend NPU")
    if any(tensor.device != query.device for tensor in input_tensors):
        raise ValueError("all tensors must be on query's device")
    if any(tensor.requires_grad for tensor in input_tensors):
        raise NotImplementedError("sparse_flash_attention is forward-only")
    all_inputs_contiguous = all(tensor.is_contiguous() for tensor in input_tensors)
    if any(tensor.numel() == 0 for tensor in input_tensors):
        raise ValueError("provided tensors must be nonempty")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise NotImplementedError("floating inputs must be FP16 or BF16")
    if any(
        tensor.dtype != query.dtype for tensor in (key, value, query_rope, key_rope)
    ):
        raise ValueError("all floating inputs must have the same dtype")
    metadata_tensors = (
        sparse_indices,
        block_table,
        actual_seq_lengths_query,
        actual_seq_lengths_kv,
    )
    if any(
        tensor is not None and tensor.dtype != torch.int32
        for tensor in metadata_tensors
    ):
        raise NotImplementedError("indices, page tables and lengths must be int32")
    if (
        value.shape != key.shape
        or value.stride() != key.stride()
        or value.data_ptr() != key.data_ptr()
    ):
        raise NotImplementedError("shared-KV MLA requires value to alias key")
    if (
        query.ndim
        != (cfg.TND_TENSOR_RANK if layout_query == "TND" else cfg.BATCHED_TENSOR_RANK)
        or query.shape[-1] != cfg.KV_DIM
    ):
        raise ValueError("query rank must match its layout, with main dimension 512")
    num_query_heads = query.shape[-2]
    if num_query_heads not in cfg.SUPPORTED_QUERY_HEADS:
        raise NotImplementedError("query heads must be 1,2,4,8,16,32,64 or 128")
    if query_rope.shape != (*query.shape[:-1], cfg.ROPE_DIM):
        raise ValueError("query_rope must match query with dimension 64")
    if key.ndim != (
        cfg.TND_TENSOR_RANK if layout_kv == "TND" else cfg.BATCHED_TENSOR_RANK
    ) or key.shape[-2:] != (1, cfg.KV_DIM):
        raise ValueError(
            "cache rank must match its layout, with one KV head and dimension 512"
        )
    if key_rope.shape != (*key.shape[:-1], cfg.ROPE_DIM):
        raise ValueError("key_rope must match key with dimension 64")
    if layout_query == "TND":
        if actual_seq_lengths_query is None or actual_seq_lengths_query.ndim != 1:
            raise ValueError("TND query requires cumulative query lengths")
        num_requests = actual_seq_lengths_query.numel()
        query_seq_capacity = 0
        num_query_tokens = query.shape[0]
    else:
        num_requests, query_seq_capacity = query.shape[:2]
        num_query_tokens = num_requests * query_seq_capacity
    for lengths in (actual_seq_lengths_query, actual_seq_lengths_kv):
        if lengths is not None and lengths.shape != (num_requests,):
            raise ValueError("length tensors must have shape [batch_size]")
    if (
        sparse_indices.ndim != query.ndim
        or sparse_indices.shape[:-2] != query.shape[:-2]
        or sparse_indices.shape[-2] != 1
    ):
        raise ValueError("sparse_indices must match query tokens with one KV head")
    num_sparse_blocks = sparse_indices.shape[-1]
    paged = layout_kv == "PA_BSND"
    page_size, pages_per_request, kv_seq_capacity = (0, 0, 0)
    if paged:
        page_size = key.shape[1]
        if (
            page_size > cfg.MAX_SUPPORTED_PAGE_SIZE
            or page_size % cfg.PAGE_SIZE_ALIGNMENT
            or page_size % sparse_block_size
        ):
            raise ValueError(
                "page size must be 16-aligned, <=1024 and divisible by sparse_block_size"
            )
        if (
            block_table is None
            or block_table.ndim != 2
            or block_table.shape[0] != num_requests
        ):
            raise ValueError(
                "paged KV requires block_table [batch_size,pages_per_request]"
            )
        pages_per_request = block_table.shape[1]
    else:
        if block_table is not None:
            raise ValueError("block_table is only used with paged KV")
        if layout_kv == "BSND":
            if key.shape[0] != num_requests:
                raise ValueError("query and cache batch sizes must match")
            kv_seq_capacity = key.shape[1]
    if layout_kv in ("TND", "PA_BSND") and actual_seq_lengths_kv is None:
        raise ValueError("TND/paged KV requires actual_seq_lengths_kv")
    if not isinstance(scale_value, (int, float)) or not math.isfinite(scale_value):
        raise ValueError("scale_value must be a finite scalar")
    return _SFAContext(
        query=query,
        key=key,
        query_rope=query_rope,
        key_rope=key_rope,
        sparse_indices=sparse_indices,
        block_table=block_table,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        scale_value=scale_value,
        layout_query=layout_query,
        layout_kv=layout_kv,
        sparse_block_size=sparse_block_size,
        sparse_mode=sparse_mode,
        return_softmax_lse=return_softmax_lse,
        num_requests=num_requests,
        num_query_tokens=num_query_tokens,
        num_query_heads=num_query_heads,
        query_seq_capacity=query_seq_capacity,
        kv_seq_capacity=kv_seq_capacity,
        num_sparse_blocks=num_sparse_blocks,
        paged=paged,
        page_size=page_size,
        pages_per_request=pages_per_request,
        all_inputs_contiguous=all_inputs_contiguous,
    )


def _normalize_inputs(ctx):
    """Materialize only Q/metadata views that require contiguous storage."""
    if ctx.all_inputs_contiguous:
        return ctx
    return replace(
        ctx,
        query=_sfa_make_contiguous(ctx.query),
        query_rope=_sfa_make_contiguous(ctx.query_rope),
        sparse_indices=_sfa_make_contiguous(ctx.sparse_indices),
        block_table=_sfa_make_contiguous(ctx.block_table),
        actual_seq_lengths_query=_sfa_make_contiguous(ctx.actual_seq_lengths_query),
        actual_seq_lengths_kv=_sfa_make_contiguous(ctx.actual_seq_lengths_kv),
    )


def _empty(ctx, shape, dtype):
    return torch.empty(shape, dtype=dtype, device=ctx.query.device)


def _workspace(ctx, shape, dtype, relative_address_offset_bytes):
    "Preserve the existing scratch address placement, padding and allocation order."
    if ctx.num_query_tokens < cfg.LARGE_QUERY_BATCH_THRESHOLD:
        return _empty(ctx, shape, dtype)
    address_period_bytes = cfg.SCRATCH_ADDRESS_PERIOD_BYTES
    element_size_bytes = (
        cfg.FP32_ELEMENT_BYTES
        if dtype == torch.float32
        else cfg.FP16_BF16_ELEMENT_BYTES
    )
    num_elements = math.prod(shape)
    workspace_storage = _empty(
        ctx, (num_elements + address_period_bytes // element_size_bytes,), dtype
    )
    storage_offset_elements = (
        (
            ctx.key.data_ptr()
            + relative_address_offset_bytes
            - workspace_storage.data_ptr()
        )
        % address_period_bytes
        // element_size_bytes
    )
    contiguous_strides = []
    running_stride = 1
    for dimension_size in reversed(shape):
        contiguous_strides.insert(0, running_stride)
        running_stride *= dimension_size
    return workspace_storage.as_strided(
        shape, tuple(contiguous_strides), storage_offset_elements
    )


def _allocate_outputs(ctx):
    attention_output = _empty(ctx, ctx.query.shape, ctx.query.dtype)
    softmax_stats_shape = (
        (
            (1, ctx.num_query_tokens, ctx.num_query_heads)
            if ctx.layout_query == "TND"
            else (ctx.num_requests, 1, ctx.query_seq_capacity, ctx.num_query_heads)
        )
        if ctx.return_softmax_lse
        else (0,)
    )
    softmax_max, softmax_sum = (
        _empty(ctx, softmax_stats_shape, torch.float32),
        _empty(ctx, softmax_stats_shape, torch.float32),
    )
    return attention_output, softmax_max, softmax_sum


def _get_cache_access_spec(tensor, layout_kv):
    if tensor.is_contiguous():
        return None
    # Row IDs can overflow before stride remapping, even when an expanded
    # view has tiny physical storage. Include all groups in the logical span.
    requires_int64_element_offsets = (
        sum(
            (size - 1) * dimension_stride
            for size, dimension_stride in zip(tensor.shape, tensor.stride())
        )
        > cfg.INT32_MAX
        or math.prod(tensor.shape[:-2]) > cfg.INT32_MAX
    )
    if layout_kv == "TND":
        return (
            1,
            tensor.stride(0),
            0,
            tensor.stride(-1),
            requires_int64_element_offsets,
        )
    return (
        tensor.shape[1],
        tensor.stride(0),
        tensor.stride(1),
        tensor.stride(-1),
        requires_int64_element_offsets,
    )


def _cache_kwargs(plan):
    return dict(
        KV_ACCESS_SPEC=plan.kv_access_spec,
        KEY_ROPE_ACCESS_SPEC=plan.key_rope_access_spec,
    )


def _query_meta_kwargs(ctx):
    return dict(
        NUM_REQUESTS=ctx.num_requests,
        QUERY_SEQ_CAPACITY=ctx.query_seq_capacity,
        QUERY_IS_TND=ctx.layout_query == "TND",
        HAS_QUERY_LENGTHS=ctx.actual_seq_lengths_query is not None,
        CAUSAL=ctx.sparse_mode == cfg.SPARSE_MODE_CAUSAL,
    )


def _choose_general_split_count(head_tasks, kv_blocks, num_cube_cores):
    max_splits = min(kv_blocks, max(1, triton.cdiv(num_cube_cores, head_tasks)))
    return min(
        range(1, max_splits + 1),
        key=lambda count: (
            triton.cdiv(head_tasks * count, num_cube_cores)
            * triton.cdiv(kv_blocks, count),
            count,
        ),
    )


def _select_general_schedule(ctx, base, num_cube_cores):
    """Bound gather scratch by metadata; retain direct attention for other inputs."""
    contiguous_kv = (
        base["kv_access_spec"] is None and base["key_rope_access_spec"] is None
    )
    head_groups = triton.cdiv(ctx.num_query_heads, cfg.GENERAL_HEAD_BLOCK_SIZE)
    head_tasks = ctx.num_query_tokens * head_groups
    expanded_rows = ctx.num_sparse_blocks * ctx.sparse_block_size
    if expanded_rows >= cfg.GENERAL_MIN_GATHER_ROWS and (
        not contiguous_kv
        or head_groups > 1
        or ctx.sparse_block_size > 1
        or head_tasks >= triton.cdiv(num_cube_cores, 2)
    ):
        gathered_head_block = min(
            cfg.GENERAL_GATHERED_HEAD_BLOCK,
            max(cfg.GENERAL_HEAD_BLOCK_SIZE, ctx.num_query_heads),
        )
        gathered_head_tasks = ctx.num_query_tokens * triton.cdiv(
            ctx.num_query_heads, gathered_head_block
        )
        padded_sparse = (
            triton.cdiv(expanded_rows, cfg.SMALL_QUERY_KV_BLOCK)
            * cfg.SMALL_QUERY_KV_BLOCK
        )
        kv_blocks = padded_sparse // cfg.SMALL_QUERY_KV_BLOCK
        splits = _choose_general_split_count(
            gathered_head_tasks, kv_blocks, num_cube_cores
        )
        # All allocations in _launch_tiled_attention: KV/RoPE, mask, per-query KV end,
        # and raw FP32 partial outputs/max/sum. Dtype size is part of metadata.
        partial_bytes_per_query = (
            splits * ctx.num_query_heads * (cfg.KV_DIM + 2) * cfg.FP32_ELEMENT_BYTES
            if splits > 1
            else 0
        )
        workspace_bytes = ctx.num_query_tokens * (
            padded_sparse
            * (
                (cfg.KV_DIM + cfg.ROPE_DIM) * ctx.query.element_size()
                + cfg.FP32_ELEMENT_BYTES
            )
            + cfg.INT64_ELEMENT_BYTES
            + partial_bytes_per_query
        )
        if workspace_bytes <= cfg.GENERAL_MAX_GATHER_WORKSPACE_BYTES:
            return _SFASchedule(
                "general",
                **base,
                num_programs=min(num_cube_cores, gathered_head_tasks * splits),
                num_kv_splits=splits,
                padded_sparse=padded_sparse,
                use_general_gather=True,
                gathered_head_block=gathered_head_block,
            )
    kv_rows = (
        cfg.GENERAL_WIDE_OFFSET_KV_ROWS
        if base["generic_access_requires_int64_offsets"]
        else (
            cfg.GENERAL_CONTIGUOUS_KV_ROWS
            if contiguous_kv
            else cfg.GENERAL_STRIDED_KV_ROWS
        )
    )
    kv_blocks = triton.cdiv(expanded_rows, kv_rows)
    splits = _choose_general_split_count(head_tasks, kv_blocks, num_cube_cores)
    return _SFASchedule(
        "general",
        **base,
        num_programs=min(num_cube_cores, head_tasks * splits),
        num_kv_splits=splits,
        general_kv_rows=kv_rows,
    )


def _select_schedule(ctx, core_counts):
    "Choose a path and its launch parameters using metadata only."
    num_cube_cores, num_vector_cores = core_counts
    kv_access_spec = _get_cache_access_spec(ctx.key, ctx.layout_kv)
    key_rope_access_spec = _get_cache_access_spec(ctx.key_rope, ctx.layout_kv)
    generic_access_requires_int64_offsets = (
        kv_access_spec is not None
        and kv_access_spec[cfg.CACHE_SPEC_INT64_OFFSETS_INDEX]
        or (
            key_rope_access_spec is not None
            and key_rope_access_spec[cfg.CACHE_SPEC_INT64_OFFSETS_INDEX]
        )
    )
    contiguous_kv = kv_access_spec is None and key_rope_access_spec is None
    fast_paged_shape = (
        ctx.paged
        and ctx.num_sparse_blocks == cfg.FAST_SPARSE_ENTRIES
        and ctx.page_size == cfg.FAST_PAGE_SIZE
        and ctx.sparse_block_size == 1
    )
    use_small_head_pipeline = (
        fast_paged_shape
        and contiguous_kv
        and ctx.num_query_heads == 4
        and (2 < ctx.num_query_tokens <= cfg.SMALL_QUERY_MAX_TOKENS)
    )
    use_cross_query_pipeline = (
        fast_paged_shape
        and contiguous_kv
        and ctx.num_query_heads == 64
        and (ctx.num_query_tokens > cfg.SMALL_QUERY_MAX_TOKENS)
        and (ctx.key.numel() <= cfg.MAX_CACHE_ELEMENTS_FOR_INT32_OFFSETS)
    )
    base = dict(
        num_vector_cores=num_vector_cores,
        kv_access_spec=kv_access_spec,
        key_rope_access_spec=key_rope_access_spec,
        generic_access_requires_int64_offsets=generic_access_requires_int64_offsets,
    )
    if (
        not use_small_head_pipeline
        and ctx.paged
        and (ctx.num_query_tokens <= cfg.SMALL_QUERY_MAX_TOKENS)
        and (ctx.num_query_heads <= 64)
        and (ctx.sparse_block_size == 1)
        and (ctx.num_sparse_blocks >= cfg.SMALL_QUERY_MIN_SPARSE_ENTRIES)
        and (ctx.num_sparse_blocks <= cfg.SMALL_QUERY_MAX_SPARSE_ENTRIES)
        and contiguous_kv
    ):
        head_block, kv_block = (cfg.SMALL_QUERY_HEAD_BLOCK, cfg.SMALL_QUERY_KV_BLOCK)
        padded_sparse = triton.cdiv(ctx.num_sparse_blocks, kv_block) * kv_block
        head_groups = triton.cdiv(ctx.num_query_heads, head_block)
        splits = min(
            padded_sparse // kv_block,
            max(1, triton.cdiv(num_cube_cores, ctx.num_query_tokens * head_groups)),
        )
        if ctx.num_query_tokens > 2:
            kv_blocks = padded_sparse // kv_block
            splits = min(
                range(1, kv_blocks + 1),
                key=lambda count: (
                    triton.cdiv(
                        ctx.num_query_tokens * head_groups * count, num_cube_cores
                    )
                    * triton.cdiv(kv_blocks, count),
                    count,
                ),
            )
        programs = min(num_cube_cores, ctx.num_query_tokens * head_groups * splits)
        return _SFASchedule(
            "small",
            **base,
            num_programs=programs,
            num_kv_splits=splits,
            padded_sparse=padded_sparse,
        )
    if not (
        fast_paged_shape
        and (
            ctx.num_query_heads == cfg.FAST_QUERY_HEADS
            or use_small_head_pipeline
            or use_cross_query_pipeline
        )
    ):
        return _select_general_schedule(ctx, base, num_cube_cores)
    shared_kv_row_layout = None
    cache_storage_row_span = ctx.key.shape[0] * cfg.FAST_PAGE_SIZE
    if (
        ctx.num_query_tokens >= cfg.LARGE_QUERY_BATCH_THRESHOLD
        and ctx.key.shape[0] * cfg.FAST_PAGE_SIZE <= cfg.INT32_MAX
        and (kv_access_spec is not None or key_rope_access_spec is not None)
    ):

        def get_native_gather_row_layout(tensor, row_width):
            if (
                tensor.stride(-1) != 1
                or tensor.stride(0) % row_width
                or tensor.stride(1) % row_width
            ):
                return None
            return (
                cfg.FAST_PAGE_SIZE,
                tensor.stride(0) // row_width,
                tensor.stride(1) // row_width,
            )

        kv_row_layout, key_rope_row_layout = (
            get_native_gather_row_layout(ctx.key, cfg.KV_DIM),
            get_native_gather_row_layout(ctx.key_rope, cfg.ROPE_DIM),
        )
        if kv_row_layout is not None and kv_row_layout == key_rope_row_layout:
            storage_row_span = (
                (ctx.key.shape[0] - 1) * kv_row_layout[1]
                + cfg.FAST_PAGE_OFFSET_MASK * kv_row_layout[2]
                + 1
            )
            if storage_row_span <= cfg.INT32_MAX:
                shared_kv_row_layout, cache_storage_row_span = (
                    kv_row_layout,
                    storage_row_span,
                )
                kv_access_spec = key_rope_access_spec = None
                generic_access_requires_int64_offsets = False
    base.update(
        kv_access_spec=kv_access_spec,
        key_rope_access_spec=key_rope_access_spec,
        generic_access_requires_int64_offsets=generic_access_requires_int64_offsets,
        cache_storage_row_span=cache_storage_row_span,
        shared_kv_row_layout=shared_kv_row_layout,
    )
    if (
        ctx.num_query_tokens == 1
        or kv_access_spec is not None
        or key_rope_access_spec is not None
    ):
        num_kv_splits = cfg.SINGLE_QUERY_KV_SPLITS if ctx.num_query_tokens == 1 else 1
        num_programs = min(
            num_cube_cores,
            ctx.num_query_tokens * cfg.GATHERED_HEAD_GROUPS * num_kv_splits,
        )
        return _SFASchedule(
            "gathered", **base, num_programs=num_programs, num_kv_splits=num_kv_splits
        )
    pipeline_heads = (
        cfg.SMALL_QUERY_HEAD_BLOCK if use_small_head_pipeline else ctx.num_query_heads
    )
    num_kv_splits = (
        1
        if use_small_head_pipeline
        else (
            cfg.PIPELINE_FOUR_QUERY_SPLITS
            if ctx.num_query_tokens == 4
            else cfg.PIPELINE_EIGHT_QUERY_SPLITS if ctx.num_query_tokens == 8 else 1
        )
    )
    num_programs = min(num_cube_cores, ctx.num_query_tokens * num_kv_splits)
    workspace_padding = (
        cfg.LARGE_BATCH_WORKSPACE_PADDING
        if not (use_small_head_pipeline or use_cross_query_pipeline)
        and ctx.num_query_tokens >= cfg.LARGE_QUERY_BATCH_THRESHOLD
        else 0
    )
    use_prepared_physical_indices = (
        not use_small_head_pipeline
        and ctx.num_query_tokens >= cfg.LARGE_QUERY_BATCH_THRESHOLD
        and (ctx.key.shape[0] * cfg.FAST_PAGE_SIZE <= cfg.INT32_MAX)
    )
    return _SFASchedule(
        "pipeline",
        **base,
        num_programs=num_programs,
        num_kv_splits=num_kv_splits,
        pipeline_heads=pipeline_heads,
        workspace_padding=workspace_padding,
        use_small_head_pipeline=use_small_head_pipeline,
        use_cross_query_pipeline=use_cross_query_pipeline,
        use_prepared_physical_indices=use_prepared_physical_indices,
    )


def _launch_general(ctx, plan, outputs):
    "Run direct attention, splitting KV when query/head work underfills the device."
    if plan.use_general_gather:
        _launch_tiled_attention(ctx, plan, outputs)
        return
    split_outputs = (
        _allocate_split_outputs(ctx, plan) if plan.num_kv_splits > 1 else outputs
    )
    kv_end = (
        _empty(ctx, (ctx.num_query_tokens,), torch.int64)
        if plan.num_kv_splits > 1 and ctx.return_softmax_lse
        else ctx.kv_lengths
    )
    _sfa_general[plan.num_programs,](
        ctx.query,
        ctx.query_rope,
        ctx.key,
        ctx.key_rope,
        ctx.sparse_indices,
        ctx.block_table if ctx.paged else ctx.sparse_indices,
        ctx.query_lengths,
        ctx.kv_lengths,
        split_outputs[0],
        split_outputs[1],
        split_outputs[2],
        kv_end,
        NUM_QUERY_TOKENS=ctx.num_query_tokens,
        NUM_QUERY_HEADS=ctx.num_query_heads,
        KV_SEQ_CAPACITY=ctx.kv_seq_capacity,
        NUM_SPARSE_BLOCKS=ctx.num_sparse_blocks,
        PAGE_SIZE=ctx.page_size,
        PAGES_PER_REQUEST=ctx.pages_per_request,
        SPARSE_BLOCK_SIZE=ctx.sparse_block_size,
        KV_IS_TND=ctx.layout_kv == "TND",
        PAGED=ctx.paged,
        HAS_KV_LENGTHS=ctx.actual_seq_lengths_kv is not None,
        **_query_meta_kwargs(ctx),
        SOFTMAX_SCALE=ctx.scale_value,
        RETURN_SOFTMAX_STATS=ctx.return_softmax_lse,
        **_cache_kwargs(plan),
        HEAD_BLOCK_SIZE=cfg.GENERAL_HEAD_BLOCK_SIZE,
        KV_BLOCK_ROWS=plan.general_kv_rows,
        NUM_SPLITS=plan.num_kv_splits,
        NUM_CORES=plan.num_programs,
        limit_auto_multi_buffer_only_for_local_buffer=False,
        limit_auto_multi_buffer_of_local_buffer="no-limit",
    )

    if plan.num_kv_splits > 1:
        _launch_split_combine(ctx, plan, split_outputs, outputs, kv_end)


def _launch_tiled_attention(ctx, plan, outputs):
    "Gather once, distribute head/KV tasks, then combine their partial sums."
    gathered_kv = _empty(
        ctx, (ctx.num_query_tokens, plan.padded_sparse, cfg.KV_DIM), ctx.query.dtype
    )
    gathered_rope = _empty(
        ctx, (ctx.num_query_tokens, plan.padded_sparse, cfg.ROPE_DIM), ctx.query.dtype
    )
    mask = _empty(ctx, (ctx.num_query_tokens, plan.padded_sparse), torch.float32)
    # Omitted BSND lengths may use a capacity larger than int32, even for a
    # small expanded-stride cache. Preserve it through the reduction scratch.
    kv_end = _empty(
        ctx,
        (ctx.num_query_tokens,),
        torch.int64 if plan.use_general_gather else torch.int32,
    )
    direct_general_output = plan.use_general_gather and plan.num_kv_splits == 1
    partial_out, partial_max, partial_sum = (
        outputs if direct_general_output else _allocate_split_outputs(ctx, plan)
    )
    gather_rows = cfg.CONTIGUOUS_GATHER_ROWS
    gather_kwargs = {}
    attention_kwargs = {}
    if plan.use_general_gather:
        gather_rows = (
            cfg.WIDE_OFFSET_GATHER_ROWS
            if plan.generic_access_requires_int64_offsets
            else (
                cfg.STRIDED_GATHER_ROWS
                if plan.kv_access_spec is not None
                or plan.key_rope_access_spec is not None
                else cfg.CONTIGUOUS_GATHER_ROWS
            )
        )
        gather_kwargs = dict(
            GENERAL=True,
            SPARSE_BLOCK_SIZE=ctx.sparse_block_size,
            PAGED=ctx.paged,
            KV_IS_TND=ctx.layout_kv == "TND",
            KV_SEQ_CAPACITY=ctx.kv_seq_capacity,
            HAS_KV_LENGTHS=ctx.actual_seq_lengths_kv is not None,
            **_cache_kwargs(plan),
        )
        attention_kwargs = dict(
            KV_END=kv_end,
            NUM_SPARSE_BLOCKS=ctx.num_sparse_blocks,
            SPARSE_BLOCK_SIZE=ctx.sparse_block_size,
            RETURN_STATS=ctx.return_softmax_lse,
        )
    _sfa_gather_sparse_kv[plan.num_vector_cores,](
        ctx.key,
        ctx.key_rope,
        ctx.sparse_indices,
        ctx.block_table if ctx.paged else ctx.sparse_indices,
        gathered_kv,
        gathered_rope,
        mask,
        kv_end,
        ctx.query_lengths,
        ctx.kv_lengths,
        NUM_SPARSE_BLOCKS=ctx.num_sparse_blocks,
        PADDED_SPARSE_BLOCKS=plan.padded_sparse,
        PAGE_SIZE=ctx.page_size,
        NUM_QUERY_TOKENS=ctx.num_query_tokens,
        **_query_meta_kwargs(ctx),
        PAGES_PER_REQUEST=ctx.pages_per_request,
        GATHER_BLOCK_ROWS=gather_rows,
        NUM_CORES=plan.num_vector_cores,
        **gather_kwargs,
    )
    _sfa_tiled_attention[plan.num_programs,](
        ctx.query,
        ctx.query_rope,
        gathered_kv,
        gathered_rope,
        mask,
        partial_out,
        partial_max,
        partial_sum,
        NUM_QUERY_TOKENS=ctx.num_query_tokens,
        NUM_QUERY_HEADS=ctx.num_query_heads,
        PADDED_SPARSE_BLOCKS=plan.padded_sparse,
        HEAD_BLOCK=plan.gathered_head_block,
        KV_BLOCK=cfg.SMALL_QUERY_KV_BLOCK,
        NUM_SPLITS=plan.num_kv_splits,
        NUM_CORES=plan.num_programs,
        SOFTMAX_SCALE=ctx.scale_value,
        GENERAL=plan.use_general_gather,
        **attention_kwargs,
        limit_auto_multi_buffer_only_for_local_buffer=False,
        limit_auto_multi_buffer_of_local_buffer="no-limit",
    )
    if not direct_general_output:
        _launch_split_combine(
            ctx, plan, (partial_out, partial_max, partial_sum), outputs, kv_end
        )


def _launch_split_combine(ctx, plan, partials, outputs, kv_end):
    "Merge raw FP32 split accumulators and preserve native empty-row statistics."
    head_groups = triton.cdiv(ctx.num_query_heads, plan.gathered_head_block)
    _sfa_split_combine[ctx.num_query_tokens * head_groups,](
        *partials,
        outputs[0],
        outputs[1],
        outputs[2],
        kv_end,
        NUM_QUERY_HEADS=ctx.num_query_heads,
        NUM_SPARSE_BLOCKS=ctx.num_sparse_blocks,
        NUM_SPLITS=plan.num_kv_splits,
        HEAD_BLOCK=plan.gathered_head_block,
        RETURN_STATS=ctx.return_softmax_lse,
        GENERAL=plan.path == "general",
        SPARSE_BLOCK_SIZE=ctx.sparse_block_size,
    )


def _allocate_split_outputs(ctx, plan, outputs=None):
    """Allocate FP32 partials, or alias final stats when a path needs no combine."""
    if outputs is not None and plan.num_kv_splits == 1:
        return None, outputs[1], outputs[2]
    heads = plan.pipeline_heads or ctx.num_query_heads
    shape = (ctx.num_query_tokens * plan.num_kv_splits, heads)
    return (
        _empty(ctx, (*shape, cfg.KV_DIM), torch.float32),
        _empty(ctx, shape, torch.float32),
        _empty(ctx, shape, torch.float32),
    )


def _allocate_pipeline_workspace(ctx, plan):
    """Allocate ring buffers and one or two generations of query state, in order."""
    ring = (plan.num_programs, cfg.NUM_PIPELINE_SLOTS)
    rows = cfg.PIPELINE_KV_BLOCK_ROWS + plan.workspace_padding
    columns = cfg.KV_DIM + plan.workspace_padding
    heads = plan.pipeline_heads
    states = plan.num_programs * (2 if plan.use_cross_query_pipeline else 1)
    return _PipelineWorkspace(
        _workspace(
            ctx,
            (*ring, rows, cfg.KV_DIM),
            ctx.query.dtype,
            cfg.KV_WORKSPACE_ADDRESS_OFFSET_BYTES,
        ),
        _workspace(
            ctx,
            (*ring, rows, cfg.ROPE_DIM),
            ctx.query.dtype,
            cfg.ROPE_WORKSPACE_ADDRESS_OFFSET_BYTES,
        ),
        _workspace(
            ctx,
            (*ring, heads, rows),
            torch.float32,
            cfg.SCORE_WORKSPACE_ADDRESS_OFFSET_BYTES,
        ),
        _workspace(
            ctx,
            (*ring, heads, rows),
            ctx.query.dtype,
            cfg.WEIGHTS_WORKSPACE_ADDRESS_OFFSET_BYTES,
        ),
        _workspace(
            ctx,
            (*ring, heads, columns),
            torch.float32,
            cfg.PV_WORKSPACE_ADDRESS_OFFSET_BYTES,
        ),
        _workspace(
            ctx,
            (states, cfg.SOFTMAX_STATE_ROWS + plan.workspace_padding, heads),
            torch.float32,
            cfg.SOFTMAX_WORKSPACE_ADDRESS_OFFSET_BYTES,
        ),
        _workspace(
            ctx,
            (states, heads, columns),
            torch.float32,
            cfg.OUTPUT_WORKSPACE_ADDRESS_OFFSET_BYTES,
        ),
    )


def _prepare_pipeline_indices(ctx, plan):
    "Prepare physical rows and per-query validity before the fused pipeline."
    if not plan.use_prepared_physical_indices:
        return None, None
    physical_row_indices = _empty(ctx, ctx.sparse_indices.shape, torch.int32)
    packed_query_meta = _empty(ctx, (ctx.num_query_tokens,), torch.int32)
    prepare_indices_kwargs = (
        dict(
            ALLOW_EXACT_SORT=not plan.use_cross_query_pipeline
            and plan.cache_storage_row_span <= cfg.FP32_EXACT_INT_LIMIT
        )
        if plan.shared_kv_row_layout is None
        else dict(SHARED_KV_ROW_LAYOUT=plan.shared_kv_row_layout)
    )
    _sfa_prepare_indices[min(plan.num_vector_cores, ctx.num_query_tokens),](
        ctx.sparse_indices,
        ctx.block_table,
        ctx.query_lengths,
        ctx.kv_lengths,
        physical_row_indices,
        packed_query_meta,
        NUM_QUERY_TOKENS=ctx.num_query_tokens,
        PAGES_PER_REQUEST=ctx.pages_per_request,
        NUM_CORES=min(plan.num_vector_cores, ctx.num_query_tokens),
        **prepare_indices_kwargs,
        **_query_meta_kwargs(ctx),
    )
    return physical_row_indices, packed_query_meta


def _combine_splits(
    ctx, plan, outputs, split_weighted_value_sum, split_softmax_max, split_softmax_sum
):
    "Keep the dedicated two-split reduction and the existing general reduction."
    if plan.num_kv_splits == 1:
        return
    if plan.num_kv_splits == cfg.COMBINE_TWO_SPLITS:
        kernel, programs_per_query = (
            _sfa_combine_two,
            cfg.COMBINE_TWO_PROGRAMS_PER_QUERY,
        )
    else:
        kernel, programs_per_query = _sfa_combine, cfg.COMBINE_PROGRAMS_PER_QUERY
    kernel[ctx.num_query_tokens * programs_per_query,](
        split_weighted_value_sum,
        split_softmax_max,
        split_softmax_sum,
        outputs[0],
        outputs[1],
        outputs[2],
        ctx.query_lengths,
        ctx.kv_lengths,
        NUM_KV_SPLITS=plan.num_kv_splits,
        RETURN_SOFTMAX_STATS=ctx.return_softmax_lse,
        **_query_meta_kwargs(ctx),
    )


def _launch_gathered(ctx, plan, outputs):
    "Run the existing H128 gather/attention path for one query or strided KV."
    additive_attention_mask = _empty(
        ctx, (ctx.num_query_tokens, cfg.FAST_SPARSE_ENTRIES), torch.float32
    )
    kv_end_exclusive = _empty(ctx, (ctx.num_query_tokens,), torch.int32)
    gathered_kv = _empty(
        ctx,
        (ctx.num_query_tokens, cfg.FAST_SPARSE_ENTRIES, cfg.KV_DIM),
        ctx.query.dtype,
    )
    gathered_key_rope = _empty(
        ctx,
        (ctx.num_query_tokens, cfg.FAST_SPARSE_ENTRIES, cfg.ROPE_DIM),
        ctx.query.dtype,
    )
    _sfa_gather_sparse_kv[plan.num_vector_cores,](
        ctx.key,
        ctx.key_rope,
        ctx.sparse_indices,
        ctx.block_table,
        gathered_kv,
        gathered_key_rope,
        additive_attention_mask,
        kv_end_exclusive,
        ctx.query_lengths,
        ctx.kv_lengths,
        NUM_QUERY_TOKENS=ctx.num_query_tokens,
        PAGES_PER_REQUEST=ctx.pages_per_request,
        **_cache_kwargs(plan),
        GATHER_BLOCK_ROWS=(
            cfg.WIDE_OFFSET_GATHER_ROWS
            if plan.generic_access_requires_int64_offsets
            else (
                cfg.STRIDED_GATHER_ROWS
                if plan.kv_access_spec is not None
                or plan.key_rope_access_spec is not None
                else cfg.CONTIGUOUS_GATHER_ROWS
            )
        ),
        NUM_CORES=plan.num_vector_cores,
        **_query_meta_kwargs(ctx),
        NUM_SPARSE_BLOCKS=cfg.FAST_SPARSE_ENTRIES,
        PADDED_SPARSE_BLOCKS=cfg.FAST_SPARSE_ENTRIES,
        PAGE_SIZE=cfg.FAST_PAGE_SIZE,
    )
    split_weighted_value_sum, split_softmax_max, split_softmax_sum = (
        _allocate_split_outputs(ctx, plan, outputs)
    )
    _sfa_gathered_attention[plan.num_programs,](
        ctx.query,
        ctx.query_rope,
        gathered_kv,
        gathered_key_rope,
        split_weighted_value_sum,
        split_softmax_max,
        split_softmax_sum,
        outputs[0],
        additive_attention_mask,
        kv_end_exclusive,
        outputs[1],
        outputs[2],
        NUM_QUERY_TOKENS=ctx.num_query_tokens,
        NUM_KV_SPLITS=plan.num_kv_splits,
        NUM_CORES=plan.num_programs,
        SOFTMAX_SCALE=ctx.scale_value,
        RETURN_SOFTMAX_STATS=ctx.return_softmax_lse,
        limit_auto_multi_buffer_only_for_local_buffer=False,
        limit_auto_multi_buffer_of_local_buffer="no-limit",
    )
    _combine_splits(
        ctx,
        plan,
        outputs,
        split_weighted_value_sum,
        split_softmax_max,
        split_softmax_sum,
    )


def _launch_pipeline(ctx, plan, outputs):
    "Launch the preserved fused pipeline with the selected layout and schedule."
    workspace = _allocate_pipeline_workspace(ctx, plan)
    split_weighted_value_sum, split_softmax_max, split_softmax_sum = (
        _allocate_split_outputs(ctx, plan, outputs)
    )
    physical_row_indices, packed_query_meta = _prepare_pipeline_indices(ctx, plan)
    # Unresolved strided KV has already taken the gathered path in _select_schedule.
    large_batch = (
        not plan.use_small_head_pipeline
        and ctx.num_query_tokens >= cfg.LARGE_QUERY_BATCH_THRESHOLD
    )
    _sfa_pipeline[plan.num_programs,](
        ctx.query,
        ctx.num_query_heads,
        plan.pipeline_heads,
        plan.use_cross_query_pipeline,
        ctx.query_rope,
        ctx.key,
        ctx.key_rope,
        ctx.sparse_indices,
        physical_row_indices,
        ctx.block_table,
        workspace.kv_workspace,
        workspace.rope_workspace,
        workspace.score_workspace,
        workspace.unnormalized_weights_workspace,
        workspace.block_weighted_value_sum_workspace,
        workspace.softmax_state_workspace,
        workspace.output_accumulator_workspace,
        outputs[0],
        split_weighted_value_sum,
        split_softmax_max,
        split_softmax_sum,
        ctx.query_lengths,
        packed_query_meta,
        ctx.kv_lengths,
        outputs[1],
        outputs[2],
        USE_PREPARED_PHYSICAL_INDICES=plan.use_prepared_physical_indices,
        CACHE_STORAGE_ROW_SPAN=plan.cache_storage_row_span,
        NUM_QUERY_TOKENS=ctx.num_query_tokens,
        NUM_SPARSE_BLOCKS=cfg.FAST_SPARSE_ENTRIES,
        PAGE_SIZE=cfg.FAST_PAGE_SIZE,
        PAGES_PER_REQUEST=ctx.pages_per_request,
        SOFTMAX_SCALE=ctx.scale_value,
        KV_BLOCK_ROWS=cfg.PIPELINE_KV_BLOCK_ROWS,
        QK_BLOCK_COLUMNS=(
            cfg.LARGE_BATCH_QK_COLUMNS if large_batch else cfg.SMALL_BATCH_QK_COLUMNS
        ),
        GATHER_BLOCK_ROWS=(
            cfg.CONTIGUOUS_GATHER_ROWS if large_batch else cfg.SMALL_BATCH_GATHER_ROWS
        ),
        # Cache-address arithmetic width, independent of the int32 input indices.
        # Preserve the large-batch schedule and conservative 2-GiB span guard.
        USE_INT32_OFFSETS=ctx.num_query_tokens >= cfg.LARGE_QUERY_BATCH_THRESHOLD
        and plan.cache_storage_row_span * cfg.KV_DIM
        <= cfg.MAX_CACHE_ELEMENTS_FOR_INT32_OFFSETS,
        SOFTMAX_BLOCK_HEADS=(
            cfg.SMALL_PIPELINE_VECTOR_HEADS
            if plan.use_small_head_pipeline
            else cfg.PIPELINE_SOFTMAX_HEADS
        ),
        VALUE_BLOCK_COLUMNS=cfg.PIPELINE_VALUE_COLUMNS,
        NUM_CORES=plan.num_programs,
        OUTPUT_BLOCK_COLUMNS=(
            cfg.LARGE_BATCH_OUTPUT_COLUMNS
            if large_batch
            else cfg.SMALL_BATCH_OUTPUT_COLUMNS
        ),
        OUTPUT_BLOCK_HEADS=(
            cfg.SMALL_PIPELINE_VECTOR_HEADS
            if plan.use_small_head_pipeline
            else (
                cfg.LARGE_BATCH_OUTPUT_HEADS
                if large_batch
                else cfg.SMALL_BATCH_OUTPUT_HEADS
            )
        ),
        WORKSPACE_PADDING=plan.workspace_padding,
        NUM_KV_SPLITS=plan.num_kv_splits,
        RETURN_SOFTMAX_STATS=ctx.return_softmax_lse,
        **_query_meta_kwargs(ctx),
        disable_auto_cv_work_space_manage=plan.use_prepared_physical_indices,
        disable_auto_inject_block_sync=True,
        multibuffer=False,
    )
    _combine_splits(
        ctx,
        plan,
        outputs,
        split_weighted_value_sum,
        split_softmax_max,
        split_softmax_sum,
    )


def sparse_flash_attention(
    query,
    key,
    value,
    sparse_indices,
    scale_value,
    *,
    block_table=None,
    actual_seq_lengths_query=None,
    actual_seq_lengths_kv=None,
    query_rope=None,
    key_rope=None,
    sparse_block_size=1,
    layout_query="BSND",
    layout_kv="BSND",
    sparse_mode=cfg.SPARSE_MODE_CAUSAL,
    pre_tokens=cfg.INT64_MAX,
    next_tokens=cfg.INT64_MAX,
    attention_mode=cfg.MLA_ABSORB_ATTENTION_MODE,
    return_softmax_lse=False,
):
    """vLLM-Ascend SFA interface for shared-KV MLA inference on Ascend 910B.

    Returns (attention_out, softmax_max, softmax_sum). The output has query's
    shape and dtype; optional max/sum are FP32, or empty tensors when disabled.
    FP16/BF16, BSND/TND queries, BSND/TND/paged caches, block sparsity and
    right-down causal masking are supported. Main/RoPE dimensions are 512/64.
    TND lengths are cumulative; other layouts use per-request lengths.

    value must alias key with the same shape/strides (MLA absorb). Independent
    V and backward are outside this shared-KV contract. Sparse indices name
    blocks, with valid entries followed by -1 padding; duplicates count again.
    Lengths, indices and page mappings must be valid for their input storage.
    Logical pages outside the per-request table are defensively masked; physical
    page IDs must still refer to valid KV storage.
    Device metadata is consumed in kernels, including for T==batch_size.
    Strided KV is gathered directly; only noncontiguous Q/metadata is materialized.
    Contiguous inputs stay zero-copy. No host length readback or Torch compute
    fallback is used.
    """
    ctx = _validate_inputs(
        query,
        key,
        value,
        sparse_indices,
        scale_value,
        block_table=block_table,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        query_rope=query_rope,
        key_rope=key_rope,
        sparse_block_size=sparse_block_size,
        layout_query=layout_query,
        layout_kv=layout_kv,
        sparse_mode=sparse_mode,
        pre_tokens=pre_tokens,
        next_tokens=next_tokens,
        attention_mode=attention_mode,
        return_softmax_lse=return_softmax_lse,
    )
    with torch.npu.device(query.device):
        ctx = _normalize_inputs(ctx)
        outputs = _allocate_outputs(ctx)
        plan = _select_schedule(ctx, _get_core_num(query.device))
        if plan.path == "small":
            _launch_tiled_attention(ctx, plan, outputs)
        elif plan.path == "general":
            _launch_general(ctx, plan, outputs)
        elif plan.path == "gathered":
            _launch_gathered(ctx, plan, outputs)
        else:
            _launch_pipeline(ctx, plan, outputs)
        return outputs

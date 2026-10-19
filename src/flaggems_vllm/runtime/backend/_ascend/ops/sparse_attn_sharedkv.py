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

"""Ascend SparseAttnSharedKV: model-facing API, dispatch and TLE kernels.

The operator computes DSV4 shared-KV attention in three modes: SWA (sliding
window over the recent original KV), CFA (window plus the contiguous compressed
KV prefix) and SCFA (window plus the per-query selected compressed KV). Q is
TND `[T, 64, 512]`, the shared KV is PA_ND `[pages, 128, 1, 512]`, and the
per-head sinks contribute softmax denominator mass without a value.

Three entry points share the kernels in this file:

* `sparse_attn_sharedkv` is the public API. It validates the inputs and selects
  one of the entries below.
* `sparse_attn_sharedkv_impl` carries the six single-sequence schedules this
  operator is validated on (S(fa|wa) decode at KV 8193, SCFA/SWA/CFA prefill at
  Q=KV=8192).
* `sparse_attn_sharedkv_graphsafe_impl` and
  `sparse_attn_sharedkv_eager_prefill_impl` are the dynamic paths used by the
  model: device-metadata Graph capture/replay for decode, and host-length
  driven chunked prefill.

Note: the pack kernels convert the shared-KV values from BF16 to FP16 while
packing, so value magnitudes must stay inside the FP16 range for full accuracy.
"""


from __future__ import annotations

import math
from functools import lru_cache
from typing import Callable, Optional, Sequence

import torch
import triton
import triton.extension.buffer.language as bl
import triton.language as tl
import triton.language.extra.cann.extension as cann_ext
from triton.language.extra.cann import libdevice as cann_libdevice
from triton.runtime.libentry import libentry

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)


_MODEL_SHORT_SINGLE_PREFILL_HEAD_BLOCK = 32


_MODEL_SHORT_SINGLE_PREFILL_BLOCK_N = 32


_MODEL_SHORT_SINGLE_PREFILL_WIDE_BLOCK_N = 64


_MODEL_SHORT_SINGLE_PREFILL_SCFA_SHORT_CMP_BLOCK_N = 8


_HEAD_DIM = 512


_METADATA_SIZE = 1024


_SPARSE_TOPK_OPTIONS = (512, 1024)


_LAST_GRAPHSAFE_WORKSPACE: tuple[torch.Tensor, ...] | None = None


_LAST_EAGER_WORKSPACE: tuple[torch.Tensor, ...] | None = None


_EAGER_WORKSPACE_CACHE: dict[
    tuple[int, int, str, tuple[int, ...], torch.dtype], torch.Tensor
] = {}


_EAGER_Q_PLAN_CACHE: dict[
    tuple[int, int], tuple[tuple[tuple[int, ...], tuple[int, ...]], torch.Tensor]
] = {}


_EAGER_PACK_PLAN_CACHE: dict[
    tuple[int, int],
    tuple[tuple[tuple[int, ...], tuple[int, ...], int, int], torch.Tensor],
] = {}


_EAGER_BATCHED_SCFA_LAUNCH_PLAN_CACHE: dict[
    tuple[int, int], tuple[tuple[object, ...], tuple[object, ...]]
] = {}


_EAGER_COMPILED_LAUNCHER_CACHE: dict[
    tuple[int, int, tuple[int, ...], tuple[object, ...]], Callable[..., object] | object
] = {}


_EAGER_COMPILED_LAUNCHER_UNAVAILABLE = object()


_Q1024_SWA_LAUNCH_PLAN_CACHE: dict[tuple[int, int], tuple[object, ...]] = {}


_Q256_SWA_LAUNCH_PLAN_CACHE: dict[tuple[int, int, int], tuple[object, ...]] = {}


_Q1024_CFA_LAUNCH_PLAN_CACHE: dict[tuple[int, int], tuple[object, ...]] = {}


_Q256_CFA_LAUNCH_PLAN_CACHE: dict[tuple[int, int, int], tuple[object, ...]] = {}


_Q256_SCFA_LAUNCH_PLAN_CACHE: dict[tuple[int, int, int], tuple[object, ...]] = {}


_MODEL_SINGLE_SCFA_SLOT_CACHE: dict[
    tuple[int, int, torch.dtype, int], tuple[list[torch.Tensor], ...]
] = {}


def _retain_eager_workspace(*tensors: torch.Tensor) -> None:
    """保活异步 eager kernel 使用的临时 GM，直到下一次调用完成入队。"""
    global _LAST_EAGER_WORKSPACE
    _LAST_EAGER_WORKSPACE = tuple(tensors)


def _eager_stream_key(device: torch.device) -> tuple[int, int]:
    device_index = device.index
    if device_index is None:
        device_index = torch.npu.current_device()
    return device_index, int(torch.npu.current_stream(device_index).npu_stream)


def _eager_raw_stream_key(device: torch.device) -> tuple[int, int]:
    # The input is already an NPU tensor, so its device is initialized. Follow
    # FlagTree's installed raw-stream strategy without constructing a Stream.
    import torch_npu

    device_index = device.index
    if device_index is None:
        device_index = torch.npu.current_device()
    getter = getattr(torch_npu._C, "_npu_getCurrentRawStreamNoWait", None)
    if not callable(getter):
        getter = getattr(torch_npu._C, "_npu_getCurrentRawStream", None)
    if callable(getter):
        return device_index, int(getter(device_index))
    return _eager_stream_key(device)


def _eager_workspace(
    tag: str,
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
    stream_key: Optional[tuple[int, int]] = None,
) -> torch.Tensor:
    """Return a stable eager workspace owned by the current NPU stream."""
    device_index, stream_handle = stream_key or _eager_stream_key(device)
    key = (device_index, stream_handle, tag, shape, dtype)
    workspace = _EAGER_WORKSPACE_CACHE.get(key)
    if workspace is None:
        workspace = torch.empty(shape, dtype=dtype, device=device)
        _EAGER_WORKSPACE_CACHE[key] = workspace
    return workspace


@triton.jit
def _build_q_plan_kernel(
    plan, cu_q, kv_lens, total_q, BATCH_SIZE: tl.constexpr, BLOCK: tl.constexpr
):
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    batch = tl.full((BLOCK,), 0, tl.int32)
    position = tl.full((BLOCK,), 0, tl.int32)
    for index in range(BATCH_SIZE):
        start = tl.load(cu_q + index)
        end = tl.load(cu_q + index + 1)
        kv_len = tl.load(kv_lens + index)
        active = (rows >= start) & (rows < end)
        batch = tl.where(active, index, batch)
        position = tl.where(active, kv_len - (end - start) + rows - start, position)
    tl.store(plan + rows * 2, batch, rows < total_q)
    tl.store(plan + rows * 2 + 1, position, rows < total_q)


_ZERO_OUTPUT_BLOCK = 8192


@triton.jit
def _zero_output_kernel(out, start, count, BLOCK: tl.constexpr):
    offsets = start + tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(out + offsets, 0, offsets < count)


def _zero_output_like(q: torch.Tensor, start: int = 0) -> torch.Tensor:
    """Allocate the output and clear only the rows no kernel will write.

    Active rows are produced by the attention kernels, so only the tail beyond
    the active prefix needs zeroing. A large block keeps the program count low,
    which dominates the cost of this pass on small shapes.
    """
    out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    total = int(out.numel())
    rows = int(q.shape[0]) if q.dim() else 0
    begin = start * (total // rows) if start and rows else 0
    if begin < total:
        _zero_output_kernel[(triton.cdiv(total - begin, _ZERO_OUTPUT_BLOCK),)](
            out, begin, total, BLOCK=_ZERO_OUTPUT_BLOCK
        )
    return out


def _eager_q_plan(
    q_lens: list[int],
    kv_lens: list[int],
    *,
    cu_seqlens_q: torch.Tensor,
    seqused_kv: torch.Tensor,
    device: torch.device,
    stream_key: Optional[tuple[int, int]] = None,
) -> torch.Tensor:
    """Cache the device-built [batch_id, absolute_q_position] row plan."""
    cache_key = stream_key or _eager_stream_key(device)
    plan_key = (tuple(q_lens), tuple(kv_lens))
    cached = _EAGER_Q_PLAN_CACHE.get(cache_key)
    if cached is not None and cached[0] == plan_key:
        return cached[1]

    total_q = sum(q_lens)
    plan = torch.empty((total_q, 2), dtype=torch.int32, device=device)
    if total_q:
        _build_q_plan_kernel[(triton.cdiv(total_q, 128),)](
            plan,
            cu_seqlens_q,
            seqused_kv,
            total_q,
            BATCH_SIZE=len(q_lens),
            BLOCK=128,
        )
    _EAGER_Q_PLAN_CACHE[cache_key] = (plan_key, plan)
    return plan


def _eager_pack_plan(
    q_lens: list[int],
    kv_lens: list[int],
    *,
    cu_seqlens_q: torch.Tensor,
    seqused_kv: torch.Tensor,
    prefix_pad: int,
    storage_count: int,
    device: torch.device,
    stream_key: Optional[tuple[int, int]] = None,
) -> torch.Tensor:
    """Cache [batch_id, logical_position] for generic batched PagePack rows."""
    cache_key = stream_key or _eager_stream_key(device)
    plan_key = (tuple(q_lens), tuple(kv_lens), prefix_pad, storage_count)
    cached = _EAGER_PACK_PLAN_CACHE.get(cache_key)
    if cached is not None and cached[0] == plan_key:
        return cached[1]

    batch_size = len(q_lens)
    batch_block = max(2, triton.next_power_of_2(batch_size))
    plan = _eager_workspace(
        "batched.pack_plan",
        (storage_count, 2),
        dtype=torch.int32,
        device=device,
        stream_key=cache_key,
    )
    constexpr_names = (
        "BATCH_SIZE",
        "BATCH_BLOCK",
        "PREFIX_PAD",
        "BLOCK_N",
    )
    constexpr_values = (batch_size, batch_block, prefix_pad, 64)
    _launch_cached_eager_kernel(
        _sparse_attn_sharedkv_eager_build_pack_plan_kernel,
        (get_num_cores("cube"),),
        (plan, cu_seqlens_q, seqused_kv, storage_count),
        constexpr_names,
        constexpr_values,
        _EAGER_BATCHED_PACK_OPTIONS,
        (plan.dtype, cu_seqlens_q.dtype, seqused_kv.dtype, *constexpr_values),
    )
    _EAGER_PACK_PLAN_CACHE[cache_key] = (plan_key, plan)
    return plan


def _launch_cached_eager_kernel(
    kernel_entry: object,
    grid: tuple[int, ...],
    runtime_args: tuple[object, ...],
    constexpr_names: tuple[str, ...],
    constexpr_values: tuple[object, ...],
    compile_options: dict[str, object],
    dispatch_key: tuple[object, ...],
) -> None:
    """Skip repeated FlagTree LibEntry argument binding after one normal launch.

    The first call always goes through LibEntry, so compilation, specialization and
    validation remain owned by Triton.  FlagTree 3.5 exposes the resulting
    CompiledKernel in LibEntry.kernel_cache; later calls with the same explicit
    device/shape/dtype key can invoke its stream-aware runner directly.  If that
    private cache interface changes, this helper permanently falls back to LibEntry
    for the affected key.
    """
    device_index = torch.npu.current_device()
    cache_key = (device_index, id(kernel_entry), grid, dispatch_key)
    cached = _EAGER_COMPILED_LAUNCHER_CACHE.get(cache_key)
    if cached is not None and cached is not _EAGER_COMPILED_LAUNCHER_UNAVAILABLE:
        cached(*(runtime_args + constexpr_values))
        return

    launch_kwargs = dict(zip(constexpr_names, constexpr_values))
    launch_kwargs.update(compile_options)
    kernel_entry[grid](*runtime_args, **launch_kwargs)
    if cached is _EAGER_COMPILED_LAUNCHER_UNAVAILABLE:
        return

    try:
        all_args = runtime_args + constexpr_values
        specialize_indices = set(kernel_entry.specialize_indices)
        do_not_specialize_indices = set(kernel_entry.do_not_specialize_indices)
        spec_args = [
            arg for index, arg in enumerate(all_args) if index in specialize_indices
        ]
        dns_args = [
            arg
            for index, arg in enumerate(all_args)
            if index in do_not_specialize_indices
        ]
        const_args = [
            arg
            for index, arg in enumerate(all_args)
            if index not in specialize_indices
            and index not in do_not_specialize_indices
        ]
        entry_key = kernel_entry.key(spec_args, dns_args, const_args)
        compiled_kernel = kernel_entry.kernel_cache[device_index][entry_key][0]
        grid3 = (grid + (1, 1))[:3]
        _EAGER_COMPILED_LAUNCHER_CACHE[cache_key] = compiled_kernel[grid3]
    except (AttributeError, KeyError, TypeError):
        _EAGER_COMPILED_LAUNCHER_CACHE[cache_key] = _EAGER_COMPILED_LAUNCHER_UNAVAILABLE


def _launch_q256_cached_eager_kernel(
    stream_key: Optional[tuple[int, int]],
    kernel_entry: object,
    grid: tuple[int, ...],
    runtime_args: tuple[object, ...],
    constexpr_names: tuple[str, ...],
    constexpr_values: tuple[object, ...],
    compile_options: dict[str, object],
    dispatch_key: tuple[object, ...],
) -> None:
    """Reuse the caller's current stream only for the screened q256 route."""
    if stream_key is not None:
        cache_key = (stream_key[0], id(kernel_entry), grid, dispatch_key)
        cached = _EAGER_COMPILED_LAUNCHER_CACHE.get(cache_key)
        if cached is not None and cached is not _EAGER_COMPILED_LAUNCHER_UNAVAILABLE:
            cached(*(runtime_args + constexpr_values), stream=stream_key[1])
            return
    _launch_cached_eager_kernel(
        kernel_entry,
        grid,
        runtime_args,
        constexpr_names,
        constexpr_values,
        compile_options,
        dispatch_key,
    )


_SCFA_STATIC_SLOT_TRAILING_NAMES = (
    "TOTAL_CMP_TOKENS",
    "CMP_BATCH_STRIDE",
    "Q_TOKEN_OFFSET",
    "Q_POSITION_OFFSET",
    "Q_COUNT",
    "Q_COUNT_BUCKET",
    "N_LIMIT",
    "NUM_WORKERS",
    "DENSE_PREFIX",
    "BATCHED",
    "PARTITIONED",
    "PARTITION_Q_COUNT",
    "PARTITION_Q_COUNT_BUCKET",
    "PARTITION_WORKERS",
    "PARTITION_BATCH_OFFSET",
    "SCORE_GUARD",
)


_SCFA_STATIC_SLOT_RUNTIME_NAMES = {
    "Q_TOKEN_OFFSET",
    "Q_POSITION_OFFSET",
    "Q_COUNT",
    "PARTITION_Q_COUNT",
}


def _scfa_static_slot_dispatch_key(
    signature: tuple[object, ...], trailing_values: tuple[object, ...]
) -> tuple[object, ...]:
    """Keep runtime scalar values out of the direct-launch cache key."""
    if len(trailing_values) != len(_SCFA_STATIC_SLOT_TRAILING_NAMES):
        raise ValueError("SCFA static-slot launcher argument count mismatch")
    return (
        signature,
        *(
            value
            for name, value in zip(_SCFA_STATIC_SLOT_TRAILING_NAMES, trailing_values)
            if name not in _SCFA_STATIC_SLOT_RUNTIME_NAMES
        ),
    )


def _batched_scfa_static_dispatch_signature(
    positional_args: tuple[object, ...],
) -> tuple[object, ...]:
    """Describe the static pipeline tensor types without scanning every slot."""
    if len(positional_args) != 34:
        raise ValueError("SCFA static-slot positional argument count mismatch")
    tensor_group_starts = (0, 1, 2, 3, 4, 8, 12, 14, 18, 20, 24, 28, 29, 30, 31, 32)
    tensors = tuple(positional_args[index] for index in tensor_group_starts)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError("SCFA static-slot tensor argument mismatch")
    divisibility = (
        _sparse_attn_sharedkv_prefill_scfa_static_slot_pipeline_kernel.divisibility
    )
    return (
        "batched",
        *(tensor.dtype for tensor in tensors),
        *(
            positional_args[index].data_ptr() % divisibility == 0
            for index in (0, 2, 31)
        ),
        type(positional_args[33]),
        float(positional_args[33]),
    )


def _single_scfa_static_dispatch_signature(
    positional_args: tuple[object, ...],
) -> tuple[object, ...]:
    """Reuse the validated compact tensor signature in the single namespace."""
    signature = _batched_scfa_static_dispatch_signature(positional_args)
    return ("single", *signature[1:])


def _present(tensor: Optional[torch.Tensor]) -> bool:
    return isinstance(tensor, torch.Tensor) and tensor.numel() > 0


def _require_int32(name: str, tensor: Optional[torch.Tensor]) -> torch.Tensor:
    if not _present(tensor):
        raise ValueError(f"{name} must be provided and cannot be empty.")
    if tensor.dtype != torch.int32:
        raise TypeError(f"{name} dtype must be torch.int32, got {tensor.dtype}.")
    return tensor


def _check_paged_kv(
    name: str,
    kv: Optional[torch.Tensor],
    table_name: str,
    table: Optional[torch.Tensor],
    *,
    batch_size: int,
    dtype: torch.dtype,
) -> None:
    if not _present(kv):
        raise ValueError(f"{name} must be provided and cannot be empty.")
    if kv.dtype != dtype:
        raise TypeError(f"{name} dtype must match q dtype {dtype}, got {kv.dtype}.")
    if kv.dim() != 4 or int(kv.shape[2]) != 1 or int(kv.shape[3]) != _HEAD_DIM:
        raise ValueError(
            f"{name} must have PA_ND shape [P, block_size, 1, {_HEAD_DIM}]."
        )
    block_size = int(kv.shape[1])
    if block_size <= 0 or block_size > 1024 or block_size % 16 != 0:
        raise ValueError(
            f"{name} block_size must be in [1, 1024] and aligned to 16, got {block_size}."
        )

    table = _require_int32(table_name, table)
    if (
        table.dim() != 2
        or int(table.shape[0]) != batch_size
        or int(table.shape[1]) <= 0
    ):
        raise ValueError(f"{table_name} must have shape [batch_size, max_num_blocks].")
    page_ids = table.detach().cpu()
    page_count = int(kv.shape[0])
    if bool(torch.any(page_ids < 0)) or bool(torch.any(page_ids >= page_count)):
        raise ValueError(
            f"{table_name} contains a physical page id outside [0, {page_count})."
        )


def _validate_sparse_attn_sharedkv_inputs(
    q: torch.Tensor,
    *,
    ori_kv: Optional[torch.Tensor],
    cmp_kv: Optional[torch.Tensor],
    ori_sparse_indices: Optional[torch.Tensor],
    cmp_sparse_indices: Optional[torch.Tensor],
    ori_block_table: Optional[torch.Tensor],
    cmp_block_table: Optional[torch.Tensor],
    cu_seqlens_q: Optional[torch.Tensor],
    seqused_kv: Optional[torch.Tensor],
    sinks: Optional[torch.Tensor],
    metadata: Optional[torch.Tensor],
    cmp_ratio: int,
    ori_mask_mode: int,
    cmp_mask_mode: int,
    ori_win_left: int,
    ori_win_right: int,
    layout_q: str,
    layout_kv: str,
) -> None:
    """校验当前 TLE 生产路径所覆盖的 AscendC host 强约束子集。"""
    if not isinstance(q, torch.Tensor) or q.numel() == 0:
        raise ValueError("q must be a non-empty torch.Tensor.")
    if q.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"q dtype must be float16 or bfloat16, got {q.dtype}.")
    if layout_q != "TND" or layout_kv != "PA_ND":
        raise NotImplementedError(
            "The optimized TLE paths require layout_q=TND and layout_kv=PA_ND."
        )
    if q.dim() != 3:
        raise ValueError(f"TND q must be rank 3, got shape {tuple(q.shape)}.")

    total_q, num_heads_q, head_dim = (int(dim) for dim in q.shape)
    if num_heads_q % 4 != 0:
        raise ValueError(f"q_head_num must be a multiple of 4, got {num_heads_q}.")
    if head_dim != _HEAD_DIM:
        raise ValueError(f"q_head_dim only supports {_HEAD_DIM}, got {head_dim}.")

    cu_seqlens_q = _require_int32("cu_seqlens_q", cu_seqlens_q)
    if cu_seqlens_q.dim() != 1 or cu_seqlens_q.numel() < 2:
        raise ValueError(
            "cu_seqlens_q must be a 1-D tensor describing at least one batch."
        )
    cu_seqlens_cpu = cu_seqlens_q.detach().cpu()
    if int(cu_seqlens_cpu[0]) != 0:
        raise ValueError("cu_seqlens_q must start at 0.")
    if bool(torch.any(cu_seqlens_cpu[1:] < cu_seqlens_cpu[:-1])):
        raise ValueError("cu_seqlens_q must be monotonically non-decreasing.")
    if bool(torch.any(cu_seqlens_cpu < 0)) or bool(torch.any(cu_seqlens_cpu > total_q)):
        raise ValueError(f"cu_seqlens_q values must be within [0, {total_q}].")
    if int(cu_seqlens_cpu[-1]) != total_q:
        raise ValueError(f"cu_seqlens_q[-1] must equal q.shape[0] ({total_q}).")
    batch_size = cu_seqlens_q.numel() - 1

    if _present(ori_sparse_indices):
        raise ValueError(
            "ori_sparse_indices must be empty in the current AscendC contract."
        )
    if ori_mask_mode != 4 or cmp_mask_mode != 3:
        raise ValueError(
            "ori_mask_mode and cmp_mask_mode must be 4 and 3 respectively."
        )
    if ori_win_left != 127 or ori_win_right != 0:
        raise ValueError(
            "ori_win_left and ori_win_right must be 127 and 0 respectively."
        )

    _check_paged_kv(
        "ori_kv",
        ori_kv,
        "ori_block_table",
        ori_block_table,
        batch_size=batch_size,
        dtype=q.dtype,
    )
    if total_q == 1 and int(ori_block_table.shape[1]) < 65:
        raise ValueError("ori_block_table must contain at least 65 pages for decode.")
    seqused_kv = _require_int32("seqused_kv", seqused_kv)
    if seqused_kv.dim() != 1 or seqused_kv.numel() != batch_size:
        raise ValueError(
            f"seqused_kv must contain one length for each of the {batch_size} batches."
        )
    seqused_kv_cpu = seqused_kv.detach().cpu()
    if bool(torch.any(seqused_kv_cpu < 0)):
        raise ValueError("seqused_kv values must be non-negative.")
    ori_capacity = int(ori_block_table.shape[1]) * int(ori_kv.shape[1])
    if bool(torch.any(seqused_kv_cpu > ori_capacity)):
        raise ValueError(
            f"seqused_kv exceeds ori_block_table capacity ({ori_capacity})."
        )

    has_cmp = _present(cmp_kv)
    has_sparse = _present(cmp_sparse_indices)
    if has_sparse and not has_cmp:
        raise ValueError("cmp_kv must be provided when cmp_sparse_indices is provided.")
    if has_cmp:
        if cmp_ratio not in (4, 128):
            raise ValueError(
                f"cmp_ratio must be 4 or 128 when cmp_kv is provided, got {cmp_ratio}."
            )
        _check_paged_kv(
            "cmp_kv",
            cmp_kv,
            "cmp_block_table",
            cmp_block_table,
            batch_size=batch_size,
            dtype=q.dtype,
        )
        cmp_capacity = int(cmp_block_table.shape[1]) * int(cmp_kv.shape[1])
        compressed_lengths = seqused_kv_cpu // int(cmp_ratio)
        if bool(torch.any(compressed_lengths > cmp_capacity)):
            raise ValueError(
                f"compressed KV length exceeds cmp_block_table capacity ({cmp_capacity})."
            )
    elif cmp_ratio != 0:
        raise ValueError(f"cmp_ratio must be 0 when cmp_kv is absent, got {cmp_ratio}.")

    if has_sparse:
        if cmp_sparse_indices.dtype != torch.int32:
            raise TypeError("cmp_sparse_indices dtype must be torch.int32.")
        if (
            cmp_sparse_indices.dim() != 3
            or int(cmp_sparse_indices.shape[0]) != total_q
            or int(cmp_sparse_indices.shape[1]) != 1
            or int(cmp_sparse_indices.shape[2]) not in _SPARSE_TOPK_OPTIONS
        ):
            raise ValueError(
                "TND cmp_sparse_indices must have shape [T, 1, 512 or 1024]."
            )

    if not _present(sinks) or sinks.dim() != 1 or sinks.numel() != num_heads_q:
        raise ValueError(f"sinks must have shape [{num_heads_q}].")
    if sinks.dtype != torch.float32:
        raise TypeError(f"sinks dtype must be torch.float32, got {sinks.dtype}.")

    metadata = _require_int32("metadata", metadata)
    if metadata.numel() != _METADATA_SIZE:
        raise ValueError(f"metadata must contain {_METADATA_SIZE} int32 elements.")


_NUM_CORES_FALLBACK = 24


@lru_cache(maxsize=3)
def get_num_cores(op_type: str = "vector") -> int:
    """Return the physical NPU core count used by Triton-Ascend launches.

    Query the device limit through the same torch_npu API the other Ascend
    ops use; fall back to the Ascend 910B default when it is unavailable.
    """
    assert op_type in [
        "vector",
        "cube",
        "mix",
    ], f"op_type {op_type} must in ['vector', 'cube', 'mix']."
    try:
        import torch_npu  # noqa: F401

        device = torch.npu.current_device()
        torch.npu.set_device(device)
        limits = torch.npu.get_device_limit(device)
    except (ImportError, AttributeError, KeyError, TypeError, RuntimeError):
        return _NUM_CORES_FALLBACK
    key = "vector_core_num" if op_type == "vector" else "cube_core_num"
    return limits[key] or _NUM_CORES_FALLBACK


def _prefill_launch_options(*, multibuffer: bool, unit_flag: bool) -> dict[str, object]:
    """六个固定验证形状收敛后的 Triton-Ascend 编译选项。"""
    return {
        "multibuffer": multibuffer,
        "unit_flag": unit_flag,
        "limit_auto_multi_buffer_only_for_local_buffer": True,
        "limit_auto_multi_buffer_of_local_buffer": "no-l0c",
        "enable_ubuf_saving": True,
    }


_EAGER_BATCHED_PACK_OPTIONS = _prefill_launch_options(multibuffer=False, unit_flag=True)


_EAGER_BATCHED_PACK_OPTIONS["optimize_epilogue"] = False


_EAGER_BATCHED_ATTENTION_OPTIONS = _prefill_launch_options(
    multibuffer=False, unit_flag=True
)


_EAGER_BATCHED_QK_OPTIONS = dict(_EAGER_BATCHED_ATTENTION_OPTIONS)


_EAGER_BATCHED_QK_OPTIONS.update(
    multibuffer=True,
    set_workspace_multibuffer=2,
    limit_auto_multi_buffer_only_for_local_buffer=False,
)


_EAGER_CHUNK_QK_OPTIONS = dict(_EAGER_BATCHED_PACK_OPTIONS)


_EAGER_CHUNK_QK_OPTIONS.update(
    multibuffer=True,
    set_workspace_multibuffer=2,
    limit_auto_multi_buffer_only_for_local_buffer=False,
)


_EAGER_CHUNK_PV_UNIT0_OPTIONS = dict(_EAGER_BATCHED_PACK_OPTIONS)


_EAGER_CHUNK_PV_UNIT0_OPTIONS["unit_flag"] = False


_EAGER_SCFA_STAGE_OPTIONS = _prefill_launch_options(multibuffer=False, unit_flag=False)


_EAGER_SCFA_PIPELINE_OPTIONS = _prefill_launch_options(
    multibuffer=True, unit_flag=False
)


@libentry()
@triton.jit
def _sparse_attn_sharedkv_prefill_cfa_staged_qk_softmax_kernel(
    ori_prob_ptr,
    cmp_prob_ptr,
    q_ptr,
    logical_ori_kv_ptr,
    logical_cmp_kv_ptr,
    sinks_ptr,
    TOTAL_Q: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ORI_BLOCK_SIZE: tl.constexpr,
    CMP_BLOCK_SIZE: tl.constexpr,
    CMP_RATIO: tl.constexpr,
    M_BLOCK: tl.constexpr,
    ORI_N_BLOCK: tl.constexpr,
    CMP_N_BLOCK: tl.constexpr,
    D_BLOCK: tl.constexpr,
    TOKEN_START: tl.constexpr,
    TOKEN_END: tl.constexpr,
    SOFTMAX_SCALE: tl.constexpr,
    FULL_ORI_WINDOW: tl.constexpr,
    TOKEN_BURST: tl.constexpr,
    QK_LOOP_UNROLL: tl.constexpr,
    USE_ROW_RECIPROCAL: tl.constexpr,
    Q_POSITION_OFFSET: tl.constexpr = 0,
    PREFIX_PAD: tl.constexpr = 0,
):
    # CFA staged MM1 + Vec1：一个 task 处理一个 token 的 64 个 Q head，
    # 同时生成 original-window 和 dense-compressed 两份概率。两部分必须
    # 共用 row_max/denominator，不能分别 softmax 后再相加。
    tl.static_assert(
        NUM_Q_HEADS == M_BLOCK, "CFA staged path expects one token worth of Q heads."
    )
    tl.static_assert(HEAD_DIM == 512, "DSV4 CFA staged path expects D=512.")
    tl.static_assert(
        ORI_N_BLOCK == ORI_BLOCK_SIZE, "CFA original window must contain 128 slots."
    )
    tl.static_assert(
        CMP_N_BLOCK >= 64,
        "DSV4 CFA compressed prefix contains at most 64 valid tokens.",
    )
    tl.static_assert(
        CMP_N_BLOCK <= CMP_BLOCK_SIZE, "Compressed tile must fit in one PA page."
    )
    tl.static_assert(
        CMP_BLOCK_SIZE == 128, "DSV4 CFA compressed PA page is 128 tokens."
    )
    tl.static_assert(HEAD_DIM % D_BLOCK == 0, "D_BLOCK must divide HEAD_DIM.")

    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)
    offs_h = tl.arange(0, M_BLOCK)
    offs_ori_n = tl.arange(0, ORI_N_BLOCK)
    offs_cmp_n = tl.arange(0, CMP_N_BLOCK)
    offs_d = tl.arange(0, D_BLOCK)
    sink = tl.load(sinks_ptr + offs_h).to(tl.float32)

    for raw_token in tl.range(
        TOKEN_START + pid,
        TOKEN_END,
        n_programs,
        loop_unroll_factor=QK_LOOP_UNROLL,
    ):
        if FULL_ORI_WINDOW and TOKEN_BURST > 1:
            local_iter = (raw_token - TOKEN_START - pid) // n_programs
            token_group = local_iter // TOKEN_BURST
            token_lane = local_iter - token_group * TOKEN_BURST
            burst_token = (
                TOKEN_START
                + token_group * n_programs * TOKEN_BURST
                + pid * TOKEN_BURST
                + token_lane
            )
            tokens_per_supergroup = n_programs * TOKEN_BURST
            burst_token_end = (
                TOKEN_START
                + ((TOKEN_END - TOKEN_START) // tokens_per_supergroup)
                * tokens_per_supergroup
            )
            token = tl.where(raw_token < burst_token_end, burst_token, raw_token)
        else:
            token = raw_token
        q_position = token + Q_POSITION_OFFSET
        if PREFIX_PAD > 0:
            window_start = token
            valid_ori = offs_ori_n >= tl.maximum(PREFIX_PAD - q_position, 0)
        elif FULL_ORI_WINDOW:
            window_start = token - (ORI_N_BLOCK - 1)
        else:
            window_start = tl.maximum(token - (ORI_N_BLOCK - 1), 0)
            window_len = token - window_start + 1
            ori_pos = window_start + offs_ori_n
            valid_ori = offs_ori_n < window_len

        ori_scores = tl.zeros((M_BLOCK, ORI_N_BLOCK), dtype=tl.float32)
        for d_base in tl.range(0, HEAD_DIM, D_BLOCK):
            q = tl.load(
                q_ptr
                + (token * NUM_Q_HEADS + offs_h[:, None]) * HEAD_DIM
                + d_base
                + offs_d[None, :]
            )
            if FULL_ORI_WINDOW or PREFIX_PAD > 0:
                ori_k_ptr = tl.make_block_ptr(
                    base=logical_ori_kv_ptr,
                    shape=(TOTAL_Q + PREFIX_PAD, HEAD_DIM),
                    strides=(HEAD_DIM, 1),
                    offsets=(window_start, d_base),
                    block_shape=(ORI_N_BLOCK, D_BLOCK),
                    order=(1, 0),
                )
                ori_k = tl.load(ori_k_ptr)
            else:
                ori_k = tl.load(
                    logical_ori_kv_ptr
                    + ori_pos[:, None] * HEAD_DIM
                    + d_base
                    + offs_d[None, :],
                    mask=valid_ori[:, None],
                    other=0.0,
                )
            ori_scores = tl.dot(q, tl.trans(ori_k), acc=ori_scores)

        cmp_scores = tl.zeros((M_BLOCK, CMP_N_BLOCK), dtype=tl.float32)
        for d_base in tl.range(0, HEAD_DIM, D_BLOCK):
            q = tl.load(
                q_ptr
                + (token * NUM_Q_HEADS + offs_h[:, None]) * HEAD_DIM
                + d_base
                + offs_d[None, :]
            )
            cmp_k_ptr = tl.make_block_ptr(
                base=logical_cmp_kv_ptr,
                shape=(CMP_N_BLOCK, HEAD_DIM),
                strides=(HEAD_DIM, 1),
                offsets=(0, d_base),
                block_shape=(CMP_N_BLOCK, D_BLOCK),
                order=(1, 0),
            )
            cmp_k = tl.load(cmp_k_ptr)
            cmp_scores = tl.dot(q, tl.trans(cmp_k), acc=cmp_scores)

        ori_scores *= SOFTMAX_SCALE
        if not FULL_ORI_WINDOW or PREFIX_PAD > 0:
            ori_scores = tl.where(valid_ori[None, :], ori_scores, -1e6)

        cmp_threshold = (q_position + 1) // CMP_RATIO
        valid_cmp = offs_cmp_n < cmp_threshold
        cmp_scores *= SOFTMAX_SCALE
        cmp_scores = tl.where(valid_cmp[None, :], cmp_scores, -1e6)

        row_max = tl.maximum(
            sink, tl.maximum(tl.max(ori_scores, axis=1), tl.max(cmp_scores, axis=1))
        )
        ori_scores = tl.exp(ori_scores - row_max[:, None])
        if not FULL_ORI_WINDOW or PREFIX_PAD > 0:
            ori_scores = tl.where(valid_ori[None, :], ori_scores, 0.0)
        cmp_scores = tl.exp(cmp_scores - row_max[:, None])
        cmp_scores = tl.where(valid_cmp[None, :], cmp_scores, 0.0)
        denom = (
            tl.exp(sink - row_max)
            + tl.sum(ori_scores, axis=1)
            + tl.sum(cmp_scores, axis=1)
        )
        if USE_ROW_RECIPROCAL:
            inv_denom = cann_libdevice.reciprocal(denom)
            ori_scores *= inv_denom[:, None]
            cmp_scores *= inv_denom[:, None]
        else:
            ori_scores /= denom[:, None]
            cmp_scores /= denom[:, None]

        tl.store(
            ori_prob_ptr
            + (token * NUM_Q_HEADS + offs_h[:, None]) * ORI_N_BLOCK
            + offs_ori_n[None, :],
            ori_scores,
        )
        tl.store(
            cmp_prob_ptr
            + (token * NUM_Q_HEADS + offs_h[:, None]) * CMP_N_BLOCK
            + offs_cmp_n[None, :],
            cmp_scores,
        )


@libentry()
@triton.jit
def _sparse_attn_sharedkv_prefill_cfa_staged_pv_kernel(
    out_ptr,
    ori_prob_ptr,
    cmp_prob_ptr,
    logical_ori_v_ptr,
    logical_cmp_v_ptr,
    TOTAL_Q: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ORI_BLOCK_SIZE: tl.constexpr,
    CMP_BLOCK_SIZE: tl.constexpr,
    M_BLOCK: tl.constexpr,
    ORI_N_BLOCK: tl.constexpr,
    CMP_N_BLOCK: tl.constexpr,
    D_BLOCK: tl.constexpr,
    TOKEN_START: tl.constexpr,
    TOKEN_END: tl.constexpr,
    FULL_ORI_WINDOW: tl.constexpr,
    TOKEN_BURST: tl.constexpr,
    PREFIX_PAD: tl.constexpr = 0,
):
    # CFA staged MM2：P_ori@V_ori 与 P_cmp@V_cmp 已使用同一个 softmax
    # denominator，两个 Cube MMAD 可以直接在 FP32 accumulator 中相加。
    tl.static_assert(
        NUM_Q_HEADS == M_BLOCK, "CFA staged PV expects one token worth of Q heads."
    )
    tl.static_assert(HEAD_DIM == 512, "DSV4 CFA staged PV expects D=512.")
    tl.static_assert(
        ORI_N_BLOCK == ORI_BLOCK_SIZE, "CFA original probability has 128 columns."
    )
    tl.static_assert(
        CMP_N_BLOCK >= 64, "CFA compressed probability must cover 64 valid columns."
    )
    tl.static_assert(
        CMP_N_BLOCK <= CMP_BLOCK_SIZE, "Compressed tile must fit in one PA page."
    )
    tl.static_assert(HEAD_DIM % D_BLOCK == 0, "D_BLOCK must divide HEAD_DIM.")

    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)
    offs_h = tl.arange(0, M_BLOCK)
    offs_ori_n = tl.arange(0, ORI_N_BLOCK)
    offs_cmp_n = tl.arange(0, CMP_N_BLOCK)
    offs_d = tl.arange(0, D_BLOCK)

    for raw_token in tl.range(
        TOKEN_START + pid,
        TOKEN_END,
        n_programs,
        loop_unroll_factor=2,
    ):
        if FULL_ORI_WINDOW and TOKEN_BURST > 1:
            local_iter = (raw_token - TOKEN_START - pid) // n_programs
            token_group = local_iter // TOKEN_BURST
            token_lane = local_iter - token_group * TOKEN_BURST
            burst_token = (
                TOKEN_START
                + token_group * n_programs * TOKEN_BURST
                + pid * TOKEN_BURST
                + token_lane
            )
            tokens_per_supergroup = n_programs * TOKEN_BURST
            burst_token_end = (
                TOKEN_START
                + ((TOKEN_END - TOKEN_START) // tokens_per_supergroup)
                * tokens_per_supergroup
            )
            token = tl.where(raw_token < burst_token_end, burst_token, raw_token)
        else:
            token = raw_token
        if PREFIX_PAD > 0:
            window_start = token
        elif FULL_ORI_WINDOW:
            window_start = token - (ORI_N_BLOCK - 1)
        else:
            window_start = tl.maximum(token - (ORI_N_BLOCK - 1), 0)
        for d_base in range(0, HEAD_DIM, D_BLOCK):
            ori_p = tl.load(
                ori_prob_ptr
                + (token * NUM_Q_HEADS + offs_h[:, None]) * ORI_N_BLOCK
                + offs_ori_n[None, :],
            )
            ori_v_ptr = tl.make_block_ptr(
                base=logical_ori_v_ptr,
                shape=(TOTAL_Q + PREFIX_PAD, HEAD_DIM),
                strides=(HEAD_DIM, 1),
                offsets=(window_start, d_base),
                block_shape=(ORI_N_BLOCK, D_BLOCK),
                order=(1, 0),
            )
            ori_v = tl.load(ori_v_ptr)
            acc = tl.dot(ori_p, ori_v)

            cmp_p = tl.load(
                cmp_prob_ptr
                + (token * NUM_Q_HEADS + offs_h[:, None]) * CMP_N_BLOCK
                + offs_cmp_n[None, :],
            )
            cmp_v_ptr = tl.make_block_ptr(
                base=logical_cmp_v_ptr,
                shape=(CMP_N_BLOCK, HEAD_DIM),
                strides=(HEAD_DIM, 1),
                offsets=(0, d_base),
                block_shape=(CMP_N_BLOCK, D_BLOCK),
                order=(1, 0),
            )
            cmp_v = tl.load(cmp_v_ptr)
            acc = tl.dot(cmp_p, cmp_v, acc)
            tl.store(
                out_ptr
                + (token * NUM_Q_HEADS + offs_h[:, None]) * HEAD_DIM
                + d_base
                + offs_d[None, :],
                acc,
            )


@libentry()
@triton.jit
def _sparse_attn_sharedkv_prefill_scfa_dense_prefix_qk_scores_kernel(
    q_ptr,
    sparse_kv_ptr,
    score_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    softmax_scale,
    Q_COUNT: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CMP_RATIO: tl.constexpr,
    MAX_CMP_TOPK: tl.constexpr,
    M_BLOCK: tl.constexpr,
    N_BLOCK: tl.constexpr,
    D_BLOCK: tl.constexpr,
):
    # 首个512-query段的compressed KV就是连续前缀。一个task处理一个token
    # 的全部64个Q head，D512按D256累加；固定扫描N=128，再用causal阈值
    # 屏蔽尚不可见的位置。
    tl.static_assert(
        NUM_Q_HEADS == M_BLOCK, "SCFA compact QK expects all Q heads in one task."
    )
    tl.static_assert(HEAD_DIM == 512, "SCFA compact QK expects D=512.")
    tl.static_assert(MAX_CMP_TOPK == 512, "SCFA compact QK expects topK=512.")
    tl.static_assert(MAX_CMP_TOPK % N_BLOCK == 0, "N_BLOCK must divide topK.")
    tl.static_assert(HEAD_DIM % D_BLOCK == 0, "D_BLOCK must divide head dim.")

    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)
    offs_h = tl.arange(0, M_BLOCK)
    offs_n = tl.arange(0, N_BLOCK)
    offs_d = tl.arange(0, D_BLOCK)

    for local_q in range(pid, Q_COUNT, n_programs):
        q_token = local_q
        cmp_threshold = (q_token + 1) // CMP_RATIO
        row_max = tl.full((M_BLOCK,), float("-inf"), dtype=tl.float32)
        row_sum = tl.zeros((M_BLOCK,), dtype=tl.float32)
        for topk_base in tl.range(0, 128, N_BLOCK):
            topk_id = topk_base + offs_n
            valid_sparse = topk_id < cmp_threshold
            scores = tl.zeros((M_BLOCK, N_BLOCK), dtype=tl.float32)
            for d_base in tl.range(0, HEAD_DIM, D_BLOCK):
                q = tl.load(
                    q_ptr
                    + (q_token * NUM_Q_HEADS + offs_h[:, None]) * HEAD_DIM
                    + d_base
                    + offs_d[None, :]
                )
                k_ptr = tl.make_block_ptr(
                    base=sparse_kv_ptr,
                    shape=(MAX_CMP_TOPK, HEAD_DIM),
                    strides=(HEAD_DIM, 1),
                    offsets=(topk_base, d_base),
                    block_shape=(N_BLOCK, D_BLOCK),
                    order=(1, 0),
                )
                k = tl.load(k_ptr)
                scores = tl.dot(q, tl.trans(k), acc=scores)
            scores *= softmax_scale
            scores = tl.where(valid_sparse[None, :], scores, -1e6)
            new_max = tl.maximum(row_max, tl.max(scores, axis=1))
            alpha = tl.exp(row_max - new_max)
            tile_p = tl.exp(scores - new_max[:, None])
            tile_p = tl.where(valid_sparse[None, :], tile_p, 0.0)
            row_sum = row_sum * alpha + tl.sum(tile_p, axis=1)
            row_max = new_max
            tl.store(
                score_ptr
                + (local_q * NUM_Q_HEADS + offs_h[:, None]) * MAX_CMP_TOPK
                + topk_id[None, :],
                scores,
            )
        row_max = tl.where(cmp_threshold > 0, row_max, 0.0)
        tl.store(partial_max_ptr + local_q * NUM_Q_HEADS + offs_h, row_max)
        tl.store(partial_sum_ptr + local_q * NUM_Q_HEADS + offs_h, row_sum)


@libentry()
@triton.jit
def _sparse_attn_sharedkv_prefill_scfa_dense_prefix_pv_partial_kernel(
    sparse_kv_ptr,
    score_ptr,
    partial_max_ptr,
    partial_acc_ptr,
    Q_COUNT: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CMP_RATIO: tl.constexpr,
    MAX_CMP_TOPK: tl.constexpr,
    M_BLOCK: tl.constexpr,
    N_BLOCK: tl.constexpr,
    D_BLOCK: tl.constexpr,
):
    # Sparse MM2：row_max/row_sum 已由 MM1 阶段生成，这里只重建未归一化
    # P 并计算 P@V partial。N64 让 FP32 score/P、V 和 D512 acc 落入 UB。
    tl.static_assert(
        NUM_Q_HEADS == M_BLOCK, "SCFA compact PV expects all Q heads in one task."
    )
    tl.static_assert(HEAD_DIM == 512, "SCFA compact PV expects D=512.")
    tl.static_assert(MAX_CMP_TOPK == 512, "SCFA compact PV expects topK=512.")
    tl.static_assert(MAX_CMP_TOPK % N_BLOCK == 0, "N_BLOCK must divide topK.")
    tl.static_assert(HEAD_DIM % D_BLOCK == 0, "D_BLOCK must divide head dim.")

    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)
    offs_h = tl.arange(0, M_BLOCK)
    offs_n = tl.arange(0, N_BLOCK)
    offs_d = tl.arange(0, D_BLOCK)

    for local_q in range(pid, Q_COUNT, n_programs):
        q_token = local_q
        cmp_threshold = (q_token + 1) // CMP_RATIO
        row_max = tl.load(partial_max_ptr + local_q * NUM_Q_HEADS + offs_h)
        for d_base in tl.range(0, HEAD_DIM, D_BLOCK):
            acc = tl.zeros((M_BLOCK, D_BLOCK), dtype=tl.float32)
            for topk_base in tl.range(0, 128, N_BLOCK):
                topk_id = topk_base + offs_n
                scores = tl.load(
                    score_ptr
                    + (local_q * NUM_Q_HEADS + offs_h[:, None]) * MAX_CMP_TOPK
                    + topk_id[None, :]
                )
                valid_sparse = topk_id < cmp_threshold
                p = tl.exp(scores - row_max[:, None])
                p = tl.where(valid_sparse[None, :], p, 0.0)
                v_ptr = tl.make_block_ptr(
                    base=sparse_kv_ptr,
                    shape=(MAX_CMP_TOPK, HEAD_DIM),
                    strides=(HEAD_DIM, 1),
                    offsets=(topk_base, d_base),
                    block_shape=(N_BLOCK, D_BLOCK),
                    order=(1, 0),
                )
                v = tl.load(v_ptr)
                acc = tl.dot(p.to(v.dtype), v, acc=acc)
            tl.store(
                partial_acc_ptr
                + (local_q * NUM_Q_HEADS + offs_h[:, None]) * HEAD_DIM
                + d_base
                + offs_d[None, :],
                acc,
            )


@libentry()
@triton.jit
def _sparse_attn_sharedkv_prefill_scfa_static_slot_pipeline_kernel(
    q_ptr,
    logical_cmp_kv_ptr,
    cmp_sparse_indices_ptr,
    q_plan_ptr,
    kv0_ptr,
    kv1_ptr,
    kv2_ptr,
    kv3_ptr,
    valid0_ptr,
    valid1_ptr,
    valid2_ptr,
    valid3_ptr,
    score0_ptr,
    score1_ptr,
    prob0_ptr,
    prob1_ptr,
    prob2_ptr,
    prob3_ptr,
    acc0_ptr,
    acc1_ptr,
    max0_ptr,
    max1_ptr,
    max2_ptr,
    max3_ptr,
    sum0_ptr,
    sum1_ptr,
    sum2_ptr,
    sum3_ptr,
    ori_acc_ptr,
    ori_max_ptr,
    ori_sum_ptr,
    sinks_ptr,
    out_ptr,
    softmax_scale,
    TOTAL_CMP_TOKENS: tl.constexpr,
    CMP_BATCH_STRIDE: tl.constexpr,
    Q_TOKEN_OFFSET,
    Q_POSITION_OFFSET,
    Q_COUNT,
    Q_COUNT_BUCKET: tl.constexpr,
    N_LIMIT: tl.constexpr,
    NUM_WORKERS: tl.constexpr,
    DENSE_PREFIX: tl.constexpr,
    BATCHED: tl.constexpr,
    PARTITIONED: tl.constexpr = False,
    PARTITION_Q_COUNT=0,
    PARTITION_Q_COUNT_BUCKET: tl.constexpr = 1,
    PARTITION_WORKERS: tl.constexpr = 0,
    PARTITION_BATCH_OFFSET: tl.constexpr = 0,
    SCORE_GUARD: tl.constexpr = True,
):
    """SCFA compressed 分支的静态槽 CV 流水。

    默认情况下，一个 program 固定负责编号为 pid、pid + NUM_WORKERS、... 的
    query token。PARTITIONED=True 时，program grid 按 batch 分区，每个 program
    只遍历所属 batch 的本地 query 时间轴，避免一个 event ring 跨 sequence 复用。
    流水包含五个阶段：V0/Pack、C1/QK、V1/softmax、C2/PV、V2/merge；当
    step=t 时，它们分别处理第 t、t-1、t-2、t-3、t-4 个本地 query。

    KV、score、probability、accumulator 和 softmax 统计量都通过显式独立的
    workspace 参数传递。外层每四步静态展开，lane 在编译期决定每个阶段使用
    哪个槽，使编译器能够证明同一轮各阶段没有地址别名，并据此安排 AIC/AIV
    并行。late query 由 V0 根据稀疏索引收集 KV；early dense-prefix 的候选是
    连续前缀，因此跳过 V0，C1/C2 直接读取 logical_cmp_kv。
    SCORE_GUARD=True 时先等待 score 槽的消费者确认再复用（模型动态路径），
    六条验证形状传 False，省略这组同步。
    """
    if SCORE_GUARD:
        for score_free_event in tl.static_range(0, 2):
            cann_ext.sync_block_set(
                "vector",
                "cube",
                score_free_event,
                cann_ext.PIPE.PIPE_MTE3,
                cann_ext.PIPE.PIPE_MTE2,
            )

    pid = tl.program_id(0)
    sub_vec = cann_ext.sub_vec_id()
    offs_h = tl.arange(0, 64)
    offs_h_vec = sub_vec * 32 + tl.arange(0, 32)

    offs_n64 = tl.arange(0, 64)
    offs_n128 = tl.arange(0, 128)
    offs_n_all = tl.arange(0, N_LIMIT)
    offs_d256 = tl.arange(0, 256)
    offs_d512 = tl.arange(0, 512)
    ring_n: tl.constexpr = 512
    if PARTITIONED:
        tl.static_assert(
            PARTITION_Q_COUNT_BUCKET > 0,
            "Partitioned SCFA requires a positive Q bucket.",
        )
        tl.static_assert(
            PARTITION_WORKERS > 0, "Partitioned SCFA requires workers per batch."
        )
        partition_index = pid // PARTITION_WORKERS
        partition_batch = PARTITION_BATCH_OFFSET + partition_index
        local_pid = pid - partition_index * PARTITION_WORKERS
        local_q_count = PARTITION_Q_COUNT
        local_q_capacity: tl.constexpr = PARTITION_Q_COUNT_BUCKET
        local_workers: tl.constexpr = PARTITION_WORKERS
    else:
        tl.static_assert(Q_COUNT_BUCKET > 0, "SCFA requires a positive Q bucket.")
        partition_batch = 0
        local_pid = pid
        local_q_count = Q_COUNT
        local_q_capacity: tl.constexpr = Q_COUNT_BUCKET
        local_workers: tl.constexpr = NUM_WORKERS
    num_steps: tl.constexpr = (local_q_capacity + local_workers - 1) // local_workers
    tl.static_assert(N_LIMIT % 128 == 0, "static-slot N_LIMIT must align to N128.")

    for kv_free_event in tl.static_range(10, 14):
        cann_ext.sync_block_set(
            "cube",
            "vector",
            kv_free_event,
            cann_ext.PIPE.PIPE_FIX,
            cann_ext.PIPE.PIPE_MTE2,
        )
    for acc_free_event in tl.static_range(14, 16):
        cann_ext.sync_block_set(
            "vector",
            "cube",
            acc_free_event,
            cann_ext.PIPE.PIPE_MTE3,
            cann_ext.PIPE.PIPE_MTE2,
        )

    for group in tl.range(0, (num_steps + 7) // 4):
        for lane in tl.static_range(0, 4):
            step = group * 4 + lane
            if lane == 0:
                pack_kv, pack_valid = kv0_ptr, valid0_ptr
                pack_kv_free_event = 10
                qk_kv, qk_valid, qk_score = kv3_ptr, valid3_ptr, score1_ptr
                v1_score, v1_valid, v1_prob = score0_ptr, valid2_ptr, prob2_ptr
                c2_prob, c2_kv, c2_acc = prob1_ptr, kv1_ptr, acc1_ptr
                c2_kv_free_event = 11
                c2_acc_free_event = 15
                v2_acc, v2_max, v2_sum = acc0_ptr, max0_ptr, sum0_ptr
                v2_acc_free_event = 14
                v1_max, v1_sum = max2_ptr, sum2_ptr
            elif lane == 1:
                pack_kv, pack_valid = kv1_ptr, valid1_ptr
                pack_kv_free_event = 11
                qk_kv, qk_valid, qk_score = kv0_ptr, valid0_ptr, score0_ptr
                v1_score, v1_valid, v1_prob = score1_ptr, valid3_ptr, prob3_ptr
                c2_prob, c2_kv, c2_acc = prob2_ptr, kv2_ptr, acc0_ptr
                c2_kv_free_event = 12
                c2_acc_free_event = 14
                v2_acc, v2_max, v2_sum = acc1_ptr, max1_ptr, sum1_ptr
                v2_acc_free_event = 15
                v1_max, v1_sum = max3_ptr, sum3_ptr
            elif lane == 2:
                pack_kv, pack_valid = kv2_ptr, valid2_ptr
                pack_kv_free_event = 12
                qk_kv, qk_valid, qk_score = kv1_ptr, valid1_ptr, score1_ptr
                v1_score, v1_valid, v1_prob = score0_ptr, valid0_ptr, prob0_ptr
                c2_prob, c2_kv, c2_acc = prob3_ptr, kv3_ptr, acc1_ptr
                c2_kv_free_event = 13
                c2_acc_free_event = 15
                v2_acc, v2_max, v2_sum = acc0_ptr, max2_ptr, sum2_ptr
                v2_acc_free_event = 14
                v1_max, v1_sum = max0_ptr, sum0_ptr
            else:
                pack_kv, pack_valid = kv3_ptr, valid3_ptr
                pack_kv_free_event = 13
                qk_kv, qk_valid, qk_score = (  # noqa: F841
                    kv2_ptr,
                    valid2_ptr,
                    score0_ptr,
                )
                v1_score, v1_valid, v1_prob = score1_ptr, valid1_ptr, prob1_ptr
                c2_prob, c2_kv, c2_acc = prob0_ptr, kv0_ptr, acc0_ptr
                c2_kv_free_event = 10
                c2_acc_free_event = 14
                v2_acc, v2_max, v2_sum = acc1_ptr, max3_ptr, sum3_ptr
                v2_acc_free_event = 15
                v1_max, v1_sum = max1_ptr, sum1_ptr

            pack_local_q = local_pid + step * local_workers
            if not DENSE_PREFIX and pack_local_q < local_q_count:
                cann_ext.sync_block_wait(
                    "cube",
                    "vector",
                    pack_kv_free_event,
                    cann_ext.PIPE.PIPE_FIX,
                    cann_ext.PIPE.PIPE_MTE2,
                )
                if PARTITIONED:
                    q_token = (
                        partition_index * PARTITION_Q_COUNT
                        + pack_local_q
                        + Q_TOKEN_OFFSET
                    )
                    source_batch = partition_batch
                    q_position = tl.load(q_plan_ptr + q_token * 2 + 1)
                else:
                    q_token = pack_local_q + Q_TOKEN_OFFSET
                if BATCHED and not PARTITIONED:
                    source_batch = tl.load(q_plan_ptr + q_token * 2)
                    q_position = tl.load(q_plan_ptr + q_token * 2 + 1)
                elif not PARTITIONED:
                    source_batch = 0
                    q_position = pack_local_q + Q_POSITION_OFFSET
                cmp_threshold = (q_position + 1) // 4
                for pair_base in range(0, 512, 128):
                    n_base = pair_base + sub_vec * 64
                    topk_id = n_base + offs_n64
                    sparse_idx = tl.load(
                        cmp_sparse_indices_ptr + q_token * 512 + topk_id
                    )
                    valid = (sparse_idx >= 0) & (sparse_idx < cmp_threshold)
                    safe_idx = tl.where(valid, sparse_idx, 0)
                    if BATCHED:
                        safe_idx += source_batch * CMP_BATCH_STRIDE
                    kv = cann_ext.index_select_simd(
                        logical_cmp_kv_ptr,
                        dim=0,
                        index=safe_idx,
                        src_shape=[TOTAL_CMP_TOKENS, 512],
                        src_offset=[-1, 0],
                        read_shape=[-1, 512],
                    )
                    tl.store(
                        pack_kv
                        + pid * 512 * 512
                        + topk_id[:, None] * 512
                        + offs_d512[None, :],
                        kv,
                    )
                    tl.store(pack_valid + pid * 512 + topk_id, valid)
                cann_ext.sync_block_set(
                    "vector",
                    "cube",
                    6,
                    cann_ext.PIPE.PIPE_MTE3,
                    cann_ext.PIPE.PIPE_MTE2,
                )

            qk_step = step - 1
            qk_local_q = local_pid + qk_step * local_workers
            if step >= 1 and qk_local_q < local_q_count:
                if SCORE_GUARD:
                    cann_ext.sync_block_wait(
                        "vector",
                        "cube",
                        (lane + 1) % 2,
                        cann_ext.PIPE.PIPE_MTE3,
                        cann_ext.PIPE.PIPE_MTE2,
                    )
                if not DENSE_PREFIX:
                    cann_ext.sync_block_wait(
                        "vector",
                        "cube",
                        6,
                        cann_ext.PIPE.PIPE_MTE3,
                        cann_ext.PIPE.PIPE_MTE2,
                    )
                q_token = qk_local_q + Q_TOKEN_OFFSET
                if PARTITIONED:
                    q_token += partition_index * PARTITION_Q_COUNT
                q_lo = tl.load(
                    q_ptr + (q_token * 64 + offs_h[:, None]) * 512 + offs_d256[None, :]
                )
                q_hi = tl.load(
                    q_ptr
                    + (q_token * 64 + offs_h[:, None]) * 512
                    + 256
                    + offs_d256[None, :]
                )
                q_l1_lo = bl.to_buffer(q_lo, cann_ext.ascend_address_space.L1)
                q_l1_hi = bl.to_buffer(q_hi, cann_ext.ascend_address_space.L1)
                for n_base in range(0, N_LIMIT, 128):
                    score = tl.zeros((64, 128), tl.float32)
                    q_tile = bl.to_tensor(q_l1_lo, writable=False)
                    if DENSE_PREFIX:
                        k0 = tl.load(
                            logical_cmp_kv_ptr
                            + partition_batch * CMP_BATCH_STRIDE * 512
                            + (n_base + offs_n128[:, None]) * 512
                            + offs_d256[None, :]
                        )
                    else:
                        k0 = tl.load(
                            qk_kv
                            + pid * 512 * 512
                            + (n_base + offs_n128[:, None]) * 512
                            + offs_d256[None, :]
                        )
                    score = tl.dot(q_tile, tl.trans(k0), acc=score)
                    q_tile = bl.to_tensor(q_l1_hi, writable=False)
                    if DENSE_PREFIX:
                        k1 = tl.load(
                            logical_cmp_kv_ptr
                            + partition_batch * CMP_BATCH_STRIDE * 512
                            + (n_base + offs_n128[:, None]) * 512
                            + 256
                            + offs_d256[None, :]
                        )
                    else:
                        k1 = tl.load(
                            qk_kv
                            + pid * 512 * 512
                            + (n_base + offs_n128[:, None]) * 512
                            + 256
                            + offs_d256[None, :]
                        )
                    score = tl.dot(q_tile, tl.trans(k1), acc=score)
                    tl.store(
                        qk_score
                        + pid * 64 * ring_n
                        + offs_h[:, None] * ring_n
                        + n_base
                        + offs_n128[None, :],
                        score,
                    )
                cann_ext.sync_block_set(
                    "cube",
                    "vector",
                    7,
                    cann_ext.PIPE.PIPE_FIX,
                    cann_ext.PIPE.PIPE_MTE2,
                )

            soft_step = step - 2
            soft_local_q = local_pid + soft_step * local_workers
            if step >= 2 and soft_local_q < local_q_count:
                cann_ext.sync_block_wait(
                    "cube",
                    "vector",
                    7,
                    cann_ext.PIPE.PIPE_FIX,
                    cann_ext.PIPE.PIPE_MTE2,
                )
                score = tl.load(
                    v1_score
                    + pid * 64 * ring_n
                    + offs_h_vec[:, None] * ring_n
                    + offs_n_all[None, :]
                ).to(tl.float32)
                if DENSE_PREFIX:
                    if PARTITIONED:
                        soft_q_token = (
                            partition_index * PARTITION_Q_COUNT
                            + soft_local_q
                            + Q_TOKEN_OFFSET
                        )
                        q_position = tl.load(q_plan_ptr + soft_q_token * 2 + 1)
                    else:
                        q_position = soft_local_q + Q_POSITION_OFFSET
                    valid = offs_n_all < (q_position + 1) // 4
                else:
                    valid = tl.load(v1_valid + pid * 512 + offs_n_all) != 0
                score = tl.where(valid[None, :], score * softmax_scale, -1e6)
                row_max = tl.max(score, axis=1)
                prob = tl.exp2((score - row_max[:, None]) * 1.4426950408889634)
                row_sum = tl.sum(prob, axis=1)
                tl.store(
                    v1_prob
                    + pid * 64 * ring_n
                    + offs_h_vec[:, None] * ring_n
                    + offs_n_all[None, :],
                    prob.to(v1_prob.dtype.element_ty),
                )
                tl.store(v1_max + pid * 64 + offs_h_vec, row_max)
                tl.store(v1_sum + pid * 64 + offs_h_vec, row_sum)
                if SCORE_GUARD:
                    cann_ext.sync_block_set(
                        "vector",
                        "cube",
                        lane % 2,
                        cann_ext.PIPE.PIPE_MTE3,
                        cann_ext.PIPE.PIPE_MTE2,
                    )
                cann_ext.sync_block_set(
                    "vector",
                    "cube",
                    8,
                    cann_ext.PIPE.PIPE_MTE3,
                    cann_ext.PIPE.PIPE_MTE2,
                )

            pv_step = step - 3
            pv_local_q = local_pid + pv_step * local_workers
            if step >= 3 and pv_local_q < local_q_count:
                cann_ext.sync_block_wait(
                    "vector",
                    "cube",
                    8,
                    cann_ext.PIPE.PIPE_MTE3,
                    cann_ext.PIPE.PIPE_MTE2,
                )
                cann_ext.sync_block_wait(
                    "vector",
                    "cube",
                    c2_acc_free_event,
                    cann_ext.PIPE.PIPE_MTE3,
                    cann_ext.PIPE.PIPE_MTE2,
                )
                acc = tl.zeros((64, 512), tl.float32)
                for n_base in range(0, N_LIMIT, 64):
                    prob = tl.load(
                        c2_prob
                        + pid * 64 * ring_n
                        + offs_h[:, None] * ring_n
                        + n_base
                        + offs_n64[None, :]
                    )
                    if DENSE_PREFIX:
                        value = tl.load(
                            logical_cmp_kv_ptr
                            + partition_batch * CMP_BATCH_STRIDE * 512
                            + (n_base + offs_n64[:, None]) * 512
                            + offs_d512[None, :]
                        )
                    else:
                        value = tl.load(
                            c2_kv
                            + pid * 512 * 512
                            + (n_base + offs_n64[:, None]) * 512
                            + offs_d512[None, :]
                        )
                    acc = tl.dot(prob, value, acc=acc)
                tl.store(
                    c2_acc
                    + pid * 64 * 512
                    + offs_h[:, None] * 512
                    + offs_d512[None, :],
                    acc,
                )
                cann_ext.sync_block_set(
                    "cube",
                    "vector",
                    9,
                    cann_ext.PIPE.PIPE_FIX,
                    cann_ext.PIPE.PIPE_MTE2,
                )
                if not DENSE_PREFIX:
                    cann_ext.sync_block_set(
                        "cube",
                        "vector",
                        c2_kv_free_event,
                        cann_ext.PIPE.PIPE_FIX,
                        cann_ext.PIPE.PIPE_MTE2,
                    )

            merge_step = step - 4
            merge_local_q = local_pid + merge_step * local_workers
            if step >= 4 and merge_local_q < local_q_count:
                cann_ext.sync_block_wait(
                    "cube",
                    "vector",
                    9,
                    cann_ext.PIPE.PIPE_FIX,
                    cann_ext.PIPE.PIPE_MTE2,
                )
                sparse_m = tl.load(v2_max + pid * 64 + offs_h_vec)
                sparse_l = tl.load(v2_sum + pid * 64 + offs_h_vec)
                q_token = merge_local_q + Q_TOKEN_OFFSET
                if PARTITIONED:
                    q_token += partition_index * PARTITION_Q_COUNT
                    ori_index = q_token
                else:
                    ori_index = merge_local_q
                ori_m = tl.load(ori_max_ptr + ori_index * 64 + offs_h_vec)
                ori_l = tl.load(ori_sum_ptr + ori_index * 64 + offs_h_vec)
                sink = tl.load(sinks_ptr + offs_h_vec).to(tl.float32)
                final_m = tl.maximum(sink, tl.maximum(ori_m, sparse_m))
                ori_scale = tl.exp2((ori_m - final_m) * 1.4426950408889634)
                sparse_scale = tl.exp2((sparse_m - final_m) * 1.4426950408889634)
                sink_scale = tl.exp2((sink - final_m) * 1.4426950408889634)
                denom = sink_scale + ori_l * ori_scale + sparse_l * sparse_scale
                for d_base in range(0, 512, 256):
                    sparse_acc = tl.load(
                        v2_acc
                        + pid * 64 * 512
                        + offs_h_vec[:, None] * 512
                        + d_base
                        + offs_d256[None, :]
                    ).to(tl.float32)
                    ori_acc = tl.load(
                        ori_acc_ptr
                        + (ori_index * 64 + offs_h_vec[:, None]) * 512
                        + d_base
                        + offs_d256[None, :]
                    ).to(tl.float32)
                    out = (
                        ori_acc * ori_scale[:, None]
                        + sparse_acc * sparse_scale[:, None]
                    ) / denom[:, None]
                    tl.store(
                        out_ptr
                        + (q_token * 64 + offs_h_vec[:, None]) * 512
                        + d_base
                        + offs_d256[None, :],
                        out.to(out_ptr.dtype.element_ty),
                    )
                cann_ext.sync_block_set(
                    "vector",
                    "cube",
                    v2_acc_free_event,
                    cann_ext.PIPE.PIPE_MTE3,
                    cann_ext.PIPE.PIPE_MTE2,
                )


@libentry()
@triton.jit
def _sparse_attn_sharedkv_prefill_scfa_merge_partials_kernel(
    ori_acc_ptr,
    ori_max_ptr,
    ori_sum_ptr,
    sparse_acc_ptr,
    sparse_max_ptr,
    sparse_sum_ptr,
    sinks_ptr,
    out_ptr,
    TOTAL_Q: tl.constexpr,
    Q_TOKEN_OFFSET,
    Q_COUNT: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
):
    # 合并 original partial、sparse partial 和 sink logit。
    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)
    head_blocks = tl.cdiv(NUM_Q_HEADS, HEAD_BLOCK)
    num_tasks = Q_COUNT * head_blocks
    offs_d = tl.arange(0, BLOCK_D)
    dim_mask = offs_d < HEAD_DIM

    for task_id in range(pid, num_tasks, n_programs):
        local_q = (task_id // head_blocks).to(tl.int64)
        q_token = local_q + Q_TOKEN_OFFSET
        head_block_id = (task_id - local_q * head_blocks).to(tl.int64)
        head_base = head_block_id * HEAD_BLOCK
        offs_h = head_base + tl.arange(0, HEAD_BLOCK).to(tl.int64)
        valid_h = offs_h < NUM_Q_HEADS
        valid_q = q_token < TOTAL_Q

        ori_m = tl.load(
            ori_max_ptr + q_token * NUM_Q_HEADS + offs_h,
            mask=valid_q & valid_h,
            other=float("-inf"),
        )
        ori_l = tl.load(
            ori_sum_ptr + q_token * NUM_Q_HEADS + offs_h,
            mask=valid_q & valid_h,
            other=0.0,
        )
        sparse_m = tl.load(
            sparse_max_ptr + local_q * NUM_Q_HEADS + offs_h,
            mask=valid_q & valid_h,
            other=float("-inf"),
        )
        sparse_l = tl.load(
            sparse_sum_ptr + local_q * NUM_Q_HEADS + offs_h,
            mask=valid_q & valid_h,
            other=0.0,
        )
        sink = tl.load(sinks_ptr + offs_h, mask=valid_h, other=0.0).to(tl.float32)

        final_m = tl.maximum(sink, tl.maximum(ori_m, sparse_m))
        sink_scale = tl.exp(sink - final_m)
        ori_scale = tl.exp(ori_m - final_m)
        sparse_scale = tl.exp(sparse_m - final_m)
        denom = sink_scale + ori_l * ori_scale + sparse_l * sparse_scale

        for d_base in tl.range(0, HEAD_DIM, BLOCK_D):
            ori_acc = tl.load(
                ori_acc_ptr
                + (q_token * NUM_Q_HEADS + offs_h[:, None]) * HEAD_DIM
                + d_base
                + offs_d[None, :],
                mask=valid_q & valid_h[:, None] & dim_mask[None, :],
                other=0.0,
            )
            sparse_acc = tl.load(
                sparse_acc_ptr
                + (local_q * NUM_Q_HEADS + offs_h[:, None]) * HEAD_DIM
                + d_base
                + offs_d[None, :],
                mask=valid_q & valid_h[:, None] & dim_mask[None, :],
                other=0.0,
            )
            sparse_acc = tl.where(sparse_l[:, None] > 0.0, sparse_acc, 0.0)
            out = (
                ori_acc * ori_scale[:, None] + sparse_acc * sparse_scale[:, None]
            ) / denom[:, None]
            tl.store(
                out_ptr
                + (q_token * NUM_Q_HEADS + offs_h[:, None]) * HEAD_DIM
                + d_base
                + offs_d[None, :],
                out.to(out_ptr.dtype.element_ty),
                mask=valid_q & valid_h[:, None] & dim_mask[None, :],
            )


@libentry()
@triton.jit
def _sparse_attn_sharedkv_prefill_swa_pack_logical_kv_kernel(
    logical_kv_ptr,
    ori_kv_ptr,
    ori_block_table_ptr,
    TOTAL_Q: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ORI_BLOCK_SIZE: tl.constexpr,
    D_BLOCK: tl.constexpr,
):
    # 把 PA_ND 的物理 page 顺序一次性还原为逻辑 token 顺序。
    #
    # AscendC 在每个 MM1/MM2 task 内用 DataCopyPA 把跨 page 的窗口片段
    # 拼成连续 L1 tile。高层 Triton 无法显式控制 L1，因此这里把相同的
    # page-table 映射前移：每个物理 page 只读一次，后续 8192 个 query
    # 的 QK/PV 都复用连续 logical_kv，避免离散 gather 被展开成逐行循环。
    tl.static_assert(
        TOTAL_Q % ORI_BLOCK_SIZE == 0, "DSV4 SWA prefill expects full logical PA pages."
    )
    tl.static_assert(HEAD_DIM % D_BLOCK == 0, "D_BLOCK must divide HEAD_DIM.")

    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)
    num_pages = TOTAL_Q // ORI_BLOCK_SIZE
    for logical_page in range(pid, num_pages, n_programs):
        physical_page = tl.load(ori_block_table_ptr + logical_page)
        for d_base in tl.range(0, HEAD_DIM, D_BLOCK):
            src = tl.make_block_ptr(
                base=ori_kv_ptr + physical_page * ORI_BLOCK_SIZE * HEAD_DIM,
                shape=(ORI_BLOCK_SIZE, HEAD_DIM),
                strides=(HEAD_DIM, 1),
                offsets=(0, d_base),
                block_shape=(ORI_BLOCK_SIZE, D_BLOCK),
                order=(1, 0),
            )
            dst = tl.make_block_ptr(
                base=logical_kv_ptr,
                shape=(TOTAL_Q, HEAD_DIM),
                strides=(HEAD_DIM, 1),
                offsets=(logical_page * ORI_BLOCK_SIZE, d_base),
                block_shape=(ORI_BLOCK_SIZE, D_BLOCK),
                order=(1, 0),
            )
            page_chunk = tl.load(src)
            tl.store(dst, page_chunk)


@libentry()
@triton.jit
def _sparse_attn_sharedkv_prefill_swa_pack_logical_k_and_fp16_v_kernel(
    logical_k_ptr,
    logical_v_ptr,
    ori_kv_ptr,
    ori_block_table_ptr,
    TOTAL_Q: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ORI_BLOCK_SIZE: tl.constexpr,
    D_BLOCK: tl.constexpr,
):
    # 同一次 PA page 读取同时生成 BF16 K 和 FP16 V。当前六个验证形状的
    # DSV4 输入值域可由 FP16 表示；转换前移后，PV 主 sweep 不再对每个重叠
    # 窗口重复执行整块 V cast。
    tl.static_assert(
        TOTAL_Q % ORI_BLOCK_SIZE == 0, "DSV4 SWA prefill expects full logical PA pages."
    )
    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)
    num_pages = TOTAL_Q // ORI_BLOCK_SIZE
    for logical_page in range(pid, num_pages, n_programs):
        physical_page = tl.load(ori_block_table_ptr + logical_page)
        for d_base in tl.range(0, HEAD_DIM, D_BLOCK):
            src = tl.make_block_ptr(
                base=ori_kv_ptr + physical_page * ORI_BLOCK_SIZE * HEAD_DIM,
                shape=(ORI_BLOCK_SIZE, HEAD_DIM),
                strides=(HEAD_DIM, 1),
                offsets=(0, d_base),
                block_shape=(ORI_BLOCK_SIZE, D_BLOCK),
                order=(1, 0),
            )
            page_chunk = tl.load(src)
            dst_k = tl.make_block_ptr(
                base=logical_k_ptr,
                shape=(TOTAL_Q, HEAD_DIM),
                strides=(HEAD_DIM, 1),
                offsets=(logical_page * ORI_BLOCK_SIZE, d_base),
                block_shape=(ORI_BLOCK_SIZE, D_BLOCK),
                order=(1, 0),
            )
            tl.store(dst_k, page_chunk)
            dst_v = tl.make_block_ptr(
                base=logical_v_ptr,
                shape=(TOTAL_Q, HEAD_DIM),
                strides=(HEAD_DIM, 1),
                offsets=(logical_page * ORI_BLOCK_SIZE, d_base),
                block_shape=(ORI_BLOCK_SIZE, D_BLOCK),
                order=(1, 0),
            )
            tl.store(dst_v, page_chunk.to(tl.float16))


@libentry()
@triton.jit
def _sparse_attn_sharedkv_chunk_pack_logical_k_and_fp16_v_kernel(
    logical_k_ptr,
    logical_v_ptr,
    paged_kv_ptr,
    block_table_ptr,
    LOGICAL_START: tl.constexpr,
    TOKEN_COUNT: tl.constexpr,
    STORAGE_COUNT: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Pack一个chunked-prefill序列实际会访问的连续KV区间。

    storage第0行对应LOGICAL_START；original路径令
    LOGICAL_START=kv_len-q_len-127，因此每个局部query i的128行窗口都从
    storage[i]开始。序列开头的负逻辑位置写0，后续QK用绝对位置mask掉。
    """
    tl.static_assert(HEAD_DIM == 512, "DSV4 chunk pack expects D=512.")
    tl.static_assert(PAGE_SIZE == 128, "DSV4 chunk pack expects page size 128.")
    tl.static_assert(BLOCK_N == 64, "Chunk pack uses one N64 tile.")
    tl.static_assert(HEAD_DIM % BLOCK_D == 0, "BLOCK_D must divide HEAD_DIM.")

    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    tile_count: tl.constexpr = (STORAGE_COUNT + BLOCK_N - 1) // BLOCK_N
    task_count: tl.constexpr = tile_count * (HEAD_DIM // BLOCK_D)

    for task_id in tl.range(pid, task_count, n_programs):
        tile_id = task_id // (HEAD_DIM // BLOCK_D)
        d_tile = task_id - tile_id * (HEAD_DIM // BLOCK_D)
        storage_row = tile_id * BLOCK_N + offs_n
        logical_pos = LOGICAL_START + storage_row
        in_storage = storage_row < STORAGE_COUNT
        valid = in_storage & (storage_row < TOKEN_COUNT) & (logical_pos >= 0)
        safe_pos = tl.maximum(logical_pos, 0)
        logical_page = safe_pos // PAGE_SIZE
        page_offset = safe_pos - logical_page * PAGE_SIZE
        valid &= logical_page < TABLE_WIDTH
        physical_page = tl.load(
            block_table_ptr + logical_page,
            mask=valid,
            other=0,
        )
        valid &= physical_page >= 0
        d_base = d_tile * BLOCK_D
        value = tl.load(
            paged_kv_ptr
            + (tl.maximum(physical_page, 0)[:, None] * PAGE_SIZE + page_offset[:, None])
            * HEAD_DIM
            + d_base
            + offs_d[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        dst = storage_row[:, None] * HEAD_DIM + d_base + offs_d[None, :]
        tl.store(logical_k_ptr + dst, value, mask=in_storage[:, None])
        tl.store(
            logical_v_ptr + dst,
            value.to(tl.float16),
            mask=in_storage[:, None],
        )


@libentry()
@triton.jit
def _sparse_attn_sharedkv_q256_fused_ori_cmp_pack_kernel(
    logical_ori_k_ptr,
    logical_ori_v_ptr,
    logical_cmp_k_ptr,
    logical_cmp_v_ptr,
    ori_kv_ptr,
    cmp_kv_ptr,
    ori_block_table_ptr,
    cmp_block_table_ptr,
    ORI_LOGICAL_START: tl.constexpr,
    ORI_TOKEN_COUNT: tl.constexpr,
    ORI_STORAGE_COUNT: tl.constexpr,
    CMP_TOKEN_COUNT: tl.constexpr,
    CMP_STORAGE_COUNT: tl.constexpr,
    ORI_TABLE_WIDTH: tl.constexpr,
    CMP_TABLE_WIDTH: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    WRITE_CMP_V: tl.constexpr,
):
    """Pack the q256 original window and compressed prefix in one launch."""
    tl.static_assert(HEAD_DIM == 512, "DSV4 q256 pack expects D=512.")
    tl.static_assert(PAGE_SIZE == 128, "DSV4 q256 pack expects N128 pages.")
    tl.static_assert(BLOCK_N == 64, "DSV4 q256 pack uses N64 tiles.")
    tl.static_assert(HEAD_DIM % BLOCK_D == 0, "BLOCK_D must divide HEAD_DIM.")

    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    d_tiles: tl.constexpr = HEAD_DIM // BLOCK_D
    ori_tiles: tl.constexpr = (ORI_STORAGE_COUNT + BLOCK_N - 1) // BLOCK_N
    cmp_tiles: tl.constexpr = (CMP_STORAGE_COUNT + BLOCK_N - 1) // BLOCK_N
    ori_tasks: tl.constexpr = ori_tiles * d_tiles
    task_count: tl.constexpr = ori_tasks + cmp_tiles * d_tiles

    for task_id in tl.range(pid, task_count, n_programs):
        if task_id < ori_tasks:
            local_task = task_id
            tile_id = local_task // d_tiles
            d_tile = local_task - tile_id * d_tiles
            storage_row = tile_id * BLOCK_N + offs_n
            logical_pos = ORI_LOGICAL_START + storage_row
            in_storage = storage_row < ORI_STORAGE_COUNT
            valid = in_storage & (storage_row < ORI_TOKEN_COUNT) & (logical_pos >= 0)
            safe_pos = tl.maximum(logical_pos, 0)
            logical_page = safe_pos // PAGE_SIZE
            page_offset = safe_pos - logical_page * PAGE_SIZE
            valid &= logical_page < ORI_TABLE_WIDTH
            physical_page = tl.load(
                ori_block_table_ptr + logical_page,
                mask=valid,
                other=0,
            )
            valid &= physical_page >= 0
            d_base = d_tile * BLOCK_D
            value = tl.load(
                ori_kv_ptr
                + (
                    tl.maximum(physical_page, 0)[:, None] * PAGE_SIZE
                    + page_offset[:, None]
                )
                * HEAD_DIM
                + d_base
                + offs_d[None, :],
                mask=valid[:, None],
                other=0.0,
            )
            dst = storage_row[:, None] * HEAD_DIM + d_base + offs_d[None, :]
            tl.store(logical_ori_k_ptr + dst, value, mask=in_storage[:, None])
            tl.store(
                logical_ori_v_ptr + dst,
                value.to(tl.float16),
                mask=in_storage[:, None],
            )
        else:
            local_task = task_id - ori_tasks
            tile_id = local_task // d_tiles
            d_tile = local_task - tile_id * d_tiles
            storage_row = tile_id * BLOCK_N + offs_n
            in_storage = storage_row < CMP_STORAGE_COUNT
            valid = in_storage & (storage_row < CMP_TOKEN_COUNT)
            logical_page = storage_row // PAGE_SIZE
            page_offset = storage_row - logical_page * PAGE_SIZE
            valid &= logical_page < CMP_TABLE_WIDTH
            physical_page = tl.load(
                cmp_block_table_ptr + logical_page,
                mask=valid,
                other=0,
            )
            valid &= physical_page >= 0
            d_base = d_tile * BLOCK_D
            value = tl.load(
                cmp_kv_ptr
                + (
                    tl.maximum(physical_page, 0)[:, None] * PAGE_SIZE
                    + page_offset[:, None]
                )
                * HEAD_DIM
                + d_base
                + offs_d[None, :],
                mask=valid[:, None],
                other=0.0,
            )
            dst = storage_row[:, None] * HEAD_DIM + d_base + offs_d[None, :]
            tl.store(logical_cmp_k_ptr + dst, value, mask=in_storage[:, None])
            if WRITE_CMP_V:
                tl.store(
                    logical_cmp_v_ptr + dst,
                    value.to(tl.float16),
                    mask=in_storage[:, None],
                )


@libentry()
@triton.jit
def _sparse_attn_sharedkv_prefill_swa_staged_qk_softmax_kernel(
    prob_ptr,
    q_ptr,
    logical_kv_ptr,
    sinks_ptr,
    TOTAL_Q: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ORI_BLOCK_SIZE: tl.constexpr,
    M_BLOCK: tl.constexpr,
    N_BLOCK: tl.constexpr,
    PROB_STRIDE: tl.constexpr,
    D_BLOCK: tl.constexpr,
    TOKEN_START: tl.constexpr,
    TOKEN_END: tl.constexpr,
    SOFTMAX_SCALE: tl.constexpr,
    FULL_WINDOW: tl.constexpr,
    TOKEN_BURST: tl.constexpr,
    QK_LOOP_UNROLL: tl.constexpr,
    USE_ROW_RECIPROCAL: tl.constexpr,
    PREFIX_PAD: tl.constexpr = 0,
    Q_POSITION_OFFSET: tl.constexpr = 0,
):
    # AscendC-like staged SWA prefill, stage 1:
    #   1. 一个 task 固定处理一个 query token 的 64 个 q-head，M_BLOCK=64；
    #   2. SWA window 固定最多 128 个 KV token，N_BLOCK=128；
    #   3. QK 的 D=512 不整体常驻：主体按 D256、前 127 行按 D128 累加；
    #   4. 本阶段只写归一化后的概率 P[M, N]，不保留 acc[M, D]。
    #
    # 这和当前 fused kernel 的关键区别是：acc[M,D] 不再和 Q/K/score/prob
    # 同时活着，从而为 M64/N128 形状释放 UB live range。
    tl.static_assert(
        NUM_Q_HEADS == M_BLOCK,
        "staged SWA prefill expects one token worth of q heads per task.",
    )
    tl.static_assert(HEAD_DIM == 512, "DSV4 staged SWA prefill expects D=512.")
    tl.static_assert(
        N_BLOCK <= ORI_BLOCK_SIZE, "N_BLOCK must not exceed one PA/SWA block."
    )
    tl.static_assert(
        PROB_STRIDE == ORI_BLOCK_SIZE,
        "prob stride follows the full 128-token SWA window.",
    )
    tl.static_assert(HEAD_DIM % D_BLOCK == 0, "D_BLOCK must divide HEAD_DIM.")

    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)
    offs_h = tl.arange(0, M_BLOCK)
    offs_n = tl.arange(0, N_BLOCK)
    offs_d = tl.arange(0, D_BLOCK)
    sink = tl.load(sinks_ptr + offs_h).to(tl.float32)
    for raw_token in tl.range(
        TOKEN_START + pid,
        TOKEN_END,
        n_programs,
        loop_unroll_factor=QK_LOOP_UNROLL,
    ):
        if (FULL_WINDOW or PREFIX_PAD > 0) and TOKEN_BURST > 1:
            local_iter = (raw_token - TOKEN_START - pid) // n_programs
            token_group = local_iter // TOKEN_BURST
            token_lane = local_iter - token_group * TOKEN_BURST
            burst_token = (
                TOKEN_START
                + token_group * n_programs * TOKEN_BURST
                + pid * TOKEN_BURST
                + token_lane
            )
            tokens_per_supergroup = n_programs * TOKEN_BURST
            burst_token_end = (
                TOKEN_START
                + ((TOKEN_END - TOKEN_START) // tokens_per_supergroup)
                * tokens_per_supergroup
            )
            token = tl.where(raw_token < burst_token_end, burst_token, raw_token)
        else:
            token = raw_token
        q_position = token + Q_POSITION_OFFSET
        if PREFIX_PAD > 0:
            window_start = token
            valid_n = offs_n >= tl.maximum(PREFIX_PAD - q_position, 0)
        elif FULL_WINDOW:
            window_start = token - (N_BLOCK - 1)
        else:
            window_start = tl.maximum(token - (N_BLOCK - 1), 0)
            window_len = token - window_start + 1
            kv_pos = window_start + offs_n
            valid_n = offs_n < window_len

        scores = tl.zeros((M_BLOCK, N_BLOCK), dtype=tl.float32)
        for d_base in tl.range(0, HEAD_DIM, D_BLOCK):
            q_block_ptr = tl.make_block_ptr(
                base=q_ptr,
                shape=(TOTAL_Q * NUM_Q_HEADS, HEAD_DIM),
                strides=(HEAD_DIM, 1),
                offsets=(token * NUM_Q_HEADS, d_base),
                block_shape=(M_BLOCK, D_BLOCK),
                order=(1, 0),
            )
            q = tl.load(q_block_ptr)
            if FULL_WINDOW or PREFIX_PAD > 0:
                k_ptr = tl.make_block_ptr(
                    base=logical_kv_ptr,
                    shape=(TOTAL_Q + PREFIX_PAD, HEAD_DIM),
                    strides=(HEAD_DIM, 1),
                    offsets=(window_start, d_base),
                    block_shape=(N_BLOCK, D_BLOCK),
                    order=(1, 0),
                )
                k = tl.load(k_ptr)
            else:
                k = tl.load(
                    logical_kv_ptr
                    + kv_pos[:, None] * HEAD_DIM
                    + d_base
                    + offs_d[None, :],
                    mask=valid_n[:, None],
                    other=0.0,
                )
            scores = tl.dot(q, tl.trans(k), acc=scores)

        if not FULL_WINDOW or PREFIX_PAD > 0:
            scores = tl.where(valid_n[None, :], scores, -1e6)

        scores *= SOFTMAX_SCALE
        row_max = tl.maximum(sink, tl.max(scores, axis=1))
        scores = tl.exp(scores - row_max[:, None])
        if not FULL_WINDOW or PREFIX_PAD > 0:
            scores = tl.where(valid_n[None, :], scores, 0.0)
        denom = tl.exp(sink - row_max) + tl.sum(scores, axis=1)
        if USE_ROW_RECIPROCAL:
            inv_denom = cann_libdevice.reciprocal(denom)
            scores *= inv_denom[:, None]
        else:
            scores = scores / denom[:, None]

        tl.store(
            prob_ptr
            + (token * NUM_Q_HEADS + offs_h[:, None]) * PROB_STRIDE
            + offs_n[None, :],
            scores,
        )


@libentry()
@triton.jit
def _sparse_attn_sharedkv_prefill_scfa_window_qk_partial_kernel(
    prob_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    q_ptr,
    logical_kv_ptr,
    TOTAL_Q: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ORI_BLOCK_SIZE: tl.constexpr,
    M_BLOCK: tl.constexpr,
    N_BLOCK: tl.constexpr,
    D_BLOCK: tl.constexpr,
    TOKEN_START: tl.constexpr,
    TOKEN_END: tl.constexpr,
    SOFTMAX_SCALE: tl.constexpr,
    FULL_WINDOW: tl.constexpr,
    TOKEN_BURST: tl.constexpr,
    Q_POSITION_OFFSET: tl.constexpr = 0,
    PREFIX_PAD: tl.constexpr = 0,
):
    # SCFA original MM1 + Vec1 partial。与 SWA staged QK 相同地按一个
    # token 的 64 head、N128、D256 计算，但不加入 sink、也不归一化 P；
    # 输出 (max, sum, exp(score-max))，供 MM2 和最终三路 merge 使用。
    tl.static_assert(
        NUM_Q_HEADS == M_BLOCK, "SCFA original QK expects all Q heads in one task."
    )
    tl.static_assert(HEAD_DIM == 512, "SCFA original QK expects D=512.")
    tl.static_assert(
        N_BLOCK == ORI_BLOCK_SIZE, "SCFA original QK follows the full SWA window."
    )
    tl.static_assert(HEAD_DIM % D_BLOCK == 0, "D_BLOCK must divide head dim.")

    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)
    offs_h = tl.arange(0, M_BLOCK)
    offs_n = tl.arange(0, N_BLOCK)
    offs_d = tl.arange(0, D_BLOCK)
    for raw_token in range(TOKEN_START + pid, TOKEN_END, n_programs):
        if FULL_WINDOW and TOKEN_BURST > 1:
            local_iter = (raw_token - TOKEN_START - pid) // n_programs
            token_group = local_iter // TOKEN_BURST
            token_lane = local_iter - token_group * TOKEN_BURST
            burst_token = (
                TOKEN_START
                + token_group * n_programs * TOKEN_BURST
                + pid * TOKEN_BURST
                + token_lane
            )
            tokens_per_supergroup = n_programs * TOKEN_BURST
            burst_token_end = (
                TOKEN_START
                + ((TOKEN_END - TOKEN_START) // tokens_per_supergroup)
                * tokens_per_supergroup
            )
            token = tl.where(raw_token < burst_token_end, burst_token, raw_token)
        else:
            token = raw_token
        q_position = token + Q_POSITION_OFFSET
        if PREFIX_PAD > 0:
            window_start = token
            valid_n = offs_n >= tl.maximum(PREFIX_PAD - q_position, 0)
            kv_pos = window_start + offs_n
        else:
            window_start = tl.maximum(token - (N_BLOCK - 1), 0)
            window_len = token - window_start + 1
            kv_pos = window_start + offs_n
            valid_n = offs_n < window_len
        scores = tl.zeros((M_BLOCK, N_BLOCK), dtype=tl.float32)
        for d_base in tl.range(0, HEAD_DIM, D_BLOCK):
            q = tl.load(
                q_ptr
                + (token * NUM_Q_HEADS + offs_h[:, None]) * HEAD_DIM
                + d_base
                + offs_d[None, :]
            )
            if FULL_WINDOW or PREFIX_PAD > 0:
                k_ptr = tl.make_block_ptr(
                    base=logical_kv_ptr,
                    shape=(TOTAL_Q + PREFIX_PAD, HEAD_DIM),
                    strides=(HEAD_DIM, 1),
                    offsets=(window_start, d_base),
                    block_shape=(N_BLOCK, D_BLOCK),
                    order=(1, 0),
                )
                k = tl.load(k_ptr)
            else:
                k = tl.load(
                    logical_kv_ptr
                    + kv_pos[:, None] * HEAD_DIM
                    + d_base
                    + offs_d[None, :],
                    mask=valid_n[:, None],
                    other=0.0,
                )
            scores += tl.dot(q, tl.trans(k))
        scores *= SOFTMAX_SCALE
        if not FULL_WINDOW or PREFIX_PAD > 0:
            scores = tl.where(valid_n[None, :], scores, -1e6)
        row_max = tl.max(scores, axis=1)
        scores = tl.exp(scores - row_max[:, None])
        if not FULL_WINDOW or PREFIX_PAD > 0:
            scores = tl.where(valid_n[None, :], scores, 0.0)
        row_sum = tl.sum(scores, axis=1)
        tl.store(
            prob_ptr
            + (token * NUM_Q_HEADS + offs_h[:, None]) * N_BLOCK
            + offs_n[None, :],
            scores,
        )
        tl.store(partial_max_ptr + token * NUM_Q_HEADS + offs_h, row_max)
        tl.store(partial_sum_ptr + token * NUM_Q_HEADS + offs_h, row_sum)


@libentry()
@triton.jit
def _sparse_attn_sharedkv_prefill_swa_staged_pv_kernel(
    out_ptr,
    prob_ptr,
    logical_kv_ptr,
    TOTAL_Q: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ORI_BLOCK_SIZE: tl.constexpr,
    M_BLOCK: tl.constexpr,
    N_BLOCK: tl.constexpr,
    PROB_STRIDE: tl.constexpr,
    D_BLOCK: tl.constexpr,
    TOKEN_START: tl.constexpr,
    TOKEN_END: tl.constexpr,
    FULL_WINDOW: tl.constexpr,
    TOKEN_BURST: tl.constexpr,
    PREFIX_PAD: tl.constexpr = 0,
):
    # AscendC-like staged SWA prefill, stage 2:
    #   P[M,128] * V[128,D_BLOCK] -> O[M,D_BLOCK]
    #
    # D_BLOCK 当前默认 512：dual-pack 已提前把 V 转成 FP16，释放了原先
    # BF16->FP16 整块 cast 占用的局部空间，因此完整 D512 MM2 可以一次
    # 完成，P[M,128] 也只读取一次。token sweep 做两路展开，让相邻 token
    # 的 MTE/MM2/Fixpipe 有机会交叠。
    tl.static_assert(
        NUM_Q_HEADS == M_BLOCK,
        "staged SWA prefill expects one token worth of q heads per task.",
    )
    tl.static_assert(HEAD_DIM == 512, "DSV4 staged SWA prefill expects D=512.")
    tl.static_assert(
        N_BLOCK <= ORI_BLOCK_SIZE, "N_BLOCK must not exceed one PA/SWA block."
    )
    tl.static_assert(
        PROB_STRIDE == ORI_BLOCK_SIZE,
        "prob stride follows the full 128-token SWA window.",
    )
    tl.static_assert(HEAD_DIM % D_BLOCK == 0, "D_BLOCK must divide HEAD_DIM.")

    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)
    offs_h = tl.arange(0, M_BLOCK)
    offs_n = tl.arange(0, N_BLOCK)
    offs_d = tl.arange(0, D_BLOCK)
    for raw_token in tl.range(
        TOKEN_START + pid,
        TOKEN_END,
        n_programs,
        loop_unroll_factor=2,
    ):
        if (FULL_WINDOW or PREFIX_PAD > 0) and TOKEN_BURST > 1:
            local_iter = (raw_token - TOKEN_START - pid) // n_programs
            token_group = local_iter // TOKEN_BURST
            token_lane = local_iter - token_group * TOKEN_BURST
            burst_token = (
                TOKEN_START
                + token_group * n_programs * TOKEN_BURST
                + pid * TOKEN_BURST
                + token_lane
            )
            tokens_per_supergroup = n_programs * TOKEN_BURST
            burst_token_end = (
                TOKEN_START
                + ((TOKEN_END - TOKEN_START) // tokens_per_supergroup)
                * tokens_per_supergroup
            )
            token = tl.where(raw_token < burst_token_end, burst_token, raw_token)
        else:
            token = raw_token
        if PREFIX_PAD > 0:
            window_start = token
        elif FULL_WINDOW:
            window_start = token - (N_BLOCK - 1)
        else:
            window_start = tl.maximum(token - (N_BLOCK - 1), 0)
            window_len = token - window_start + 1
            kv_pos = window_start + offs_n
            valid_n = offs_n < window_len

        p_ptrs = (
            prob_ptr
            + (token * NUM_Q_HEADS + offs_h[:, None]) * PROB_STRIDE
            + offs_n[None, :]
        )
        if FULL_WINDOW or PREFIX_PAD > 0:
            p = tl.load(p_ptrs)
        else:
            p = tl.load(p_ptrs, mask=valid_n[None, :], other=0.0)
        for d_base in range(0, HEAD_DIM, D_BLOCK):
            if FULL_WINDOW or PREFIX_PAD > 0:
                v_ptr = tl.make_block_ptr(
                    base=logical_kv_ptr,
                    shape=(TOTAL_Q + PREFIX_PAD, HEAD_DIM),
                    strides=(HEAD_DIM, 1),
                    offsets=(window_start, d_base),
                    block_shape=(N_BLOCK, D_BLOCK),
                    order=(1, 0),
                )
                v = tl.load(v_ptr)
            else:
                v = tl.load(
                    logical_kv_ptr
                    + kv_pos[:, None] * HEAD_DIM
                    + d_base
                    + offs_d[None, :],
                    mask=valid_n[:, None],
                    other=0.0,
                )
            out = tl.dot(p.to(tl.float16), v.to(tl.float16))
            tl.store(
                out_ptr
                + (token * NUM_Q_HEADS + offs_h[:, None]) * HEAD_DIM
                + d_base
                + offs_d[None, :],
                out,
            )


@triton.jit
def _graphsafe_q_row_threshold(
    q_row,
    cu_seqlens_q_ptr,
    seqused_kv_ptr,
    BATCH_SIZE,
    BATCH_BLOCK: tl.constexpr,
):
    """仅从 Graph 输入重建 q_row 的 batch 和 causal 右边界。"""
    active_q = tl.load(cu_seqlens_q_ptr + BATCH_SIZE).to(tl.int64)
    valid_row = q_row < active_q
    offs_b = tl.arange(0, BATCH_BLOCK)
    valid_b = offs_b < BATCH_SIZE
    q_starts = tl.load(cu_seqlens_q_ptr + offs_b, mask=valid_b, other=0).to(tl.int64)
    q_ends = tl.load(cu_seqlens_q_ptr + offs_b + 1, mask=valid_b, other=0).to(tl.int64)
    owns_row = valid_b & (q_row >= q_starts) & (q_row < q_ends)
    batch_id = tl.max(tl.where(owns_row, offs_b, -1), axis=0).to(tl.int64)
    q_start = tl.max(tl.where(owns_row, q_starts, 0), axis=0)
    q_end = tl.max(tl.where(owns_row, q_ends, 0), axis=0)
    safe_batch = tl.maximum(batch_id, 0)
    kv_len = tl.load(seqused_kv_ptr + safe_batch, mask=valid_row, other=0).to(tl.int64)
    ori_threshold = kv_len - (q_end - q_start) + (q_row - q_start) + 1
    return valid_row & (batch_id >= 0), ori_threshold, safe_batch


@triton.jit
def _graphsafe_q_row_info_i32(
    q_row,
    cu_seqlens_q_ptr,
    seqused_kv_ptr,
    BATCH_SIZE,
    BATCH_BLOCK: tl.constexpr,
):
    """pack 专用 int32 行信息，降低 N64 地址张量的 UB 占用。"""
    active_q = tl.load(cu_seqlens_q_ptr + BATCH_SIZE)
    valid_row = q_row < active_q
    offs_b = tl.arange(0, BATCH_BLOCK)
    valid_b = offs_b < BATCH_SIZE
    q_starts = tl.load(cu_seqlens_q_ptr + offs_b, mask=valid_b, other=0)
    q_ends = tl.load(cu_seqlens_q_ptr + offs_b + 1, mask=valid_b, other=0)
    owns_row = valid_b & (q_row >= q_starts) & (q_row < q_ends)
    batch_id = tl.max(tl.where(owns_row, offs_b, -1), axis=0)
    q_start = tl.max(tl.where(owns_row, q_starts, 0), axis=0)
    q_end = tl.max(tl.where(owns_row, q_ends, 0), axis=0)
    safe_batch = tl.maximum(batch_id, 0)
    kv_len = tl.load(seqused_kv_ptr + safe_batch, mask=valid_row, other=0)
    ori_threshold = kv_len - (q_end - q_start) + (q_row - q_start) + 1
    return valid_row & (batch_id >= 0), ori_threshold, safe_batch


@libentry()
@triton.jit(do_not_specialize=["TOTAL_Q", "STORAGE_COUNT", "BATCH_SIZE"])
def _sparse_attn_sharedkv_eager_batched_pack_original_kernel(
    logical_kv_ptr,
    logical_v_ptr,
    paged_kv_ptr,
    block_table_ptr,
    cu_seqlens_q_ptr,
    seqused_kv_ptr,
    TOTAL_Q,
    STORAGE_COUNT,
    BATCH_SIZE,
    BATCH_BLOCK: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PREFIX_PAD: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """一次整理多条 prefill 序列的 original KV。

    常规布局为每条序列额外保留 PREFIX_PAD=127 行。初始短 prompt 使用
    PREFIX_PAD=0，只整理真实 token，并在整个工作区末尾保留一个 N128 安全尾部。
    后续 attention 由 q_position 区分滑动后缀布局和向前 causal 布局。
    """
    tl.static_assert(HEAD_DIM == 512, "DSV4 batched pack expects D=512.")
    tl.static_assert(PAGE_SIZE == 128, "DSV4 batched pack expects page size 128.")
    tl.static_assert(BLOCK_N == 64, "Batched pack uses N64 tiles.")

    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    offs_b = tl.arange(0, BATCH_BLOCK)
    valid_b = offs_b < BATCH_SIZE
    q_starts = tl.load(cu_seqlens_q_ptr + offs_b, mask=valid_b, other=TOTAL_Q)
    storage_starts = q_starts + offs_b * PREFIX_PAD
    tiles_d: tl.constexpr = HEAD_DIM // BLOCK_D
    tile_count = (STORAGE_COUNT + BLOCK_N - 1) // BLOCK_N

    for task_id in tl.range(pid, tile_count * tiles_d, n_programs):
        tile_id = task_id // tiles_d
        d_tile = task_id - tile_id * tiles_d
        storage_row = tile_id * BLOCK_N + offs_n
        in_storage = storage_row < STORAGE_COUNT

        batch_id = (
            tl.sum(
                (
                    (storage_row[:, None] >= storage_starts[None, :]) & valid_b[None, :]
                ).to(tl.int32),
                axis=1,
            )
            - 1
        )
        valid_row = in_storage & (batch_id >= 0) & (batch_id < BATCH_SIZE)
        safe_batch = tl.maximum(batch_id, 0)
        q_start = tl.load(cu_seqlens_q_ptr + safe_batch, mask=valid_row, other=0)
        q_end = tl.load(cu_seqlens_q_ptr + safe_batch + 1, mask=valid_row, other=0)
        kv_len = tl.load(seqused_kv_ptr + safe_batch, mask=valid_row, other=0)
        local_storage = storage_row - (q_start + safe_batch * PREFIX_PAD)
        q_len = q_end - q_start
        logical_pos = kv_len - q_len - PREFIX_PAD + local_storage
        valid_row &= (
            (local_storage >= 0)
            & (local_storage < q_len + PREFIX_PAD)
            & (logical_pos >= 0)
        )

        safe_pos = tl.maximum(logical_pos, 0)
        logical_page = safe_pos // PAGE_SIZE
        page_offset = safe_pos - logical_page * PAGE_SIZE
        valid_row &= logical_page < TABLE_WIDTH
        physical_page = tl.load(
            block_table_ptr + safe_batch * TABLE_WIDTH + logical_page,
            mask=valid_row,
            other=0,
        )
        valid_row &= physical_page >= 0
        d_base = d_tile * BLOCK_D
        value = tl.load(
            paged_kv_ptr
            + (tl.maximum(physical_page, 0)[:, None] * PAGE_SIZE + page_offset[:, None])
            * HEAD_DIM
            + d_base
            + offs_d[None, :],
            mask=valid_row[:, None],
            other=0.0,
        )
        tl.store(
            logical_kv_ptr + storage_row[:, None] * HEAD_DIM + d_base + offs_d[None, :],
            value,
            mask=in_storage[:, None],
        )
        tl.store(
            logical_v_ptr + storage_row[:, None] * HEAD_DIM + d_base + offs_d[None, :],
            value.to(tl.float16),
            mask=in_storage[:, None],
        )


@libentry()
@triton.jit(do_not_specialize=["STORAGE_COUNT"])
def _sparse_attn_sharedkv_eager_build_pack_plan_kernel(
    pack_plan_ptr,
    cu_seqlens_q_ptr,
    seqused_kv_ptr,
    STORAGE_COUNT,
    BATCH_SIZE: tl.constexpr,
    BATCH_BLOCK: tl.constexpr,
    PREFIX_PAD: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Build generic PagePack row ownership once for all model layers."""
    tl.static_assert(BLOCK_N == 64, "PagePack plan uses N64 row tiles.")

    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)
    offs_n = tl.arange(0, BLOCK_N)
    offs_b = tl.arange(0, BATCH_BLOCK)
    valid_b = offs_b < BATCH_SIZE
    q_starts = tl.load(cu_seqlens_q_ptr + offs_b, mask=valid_b, other=STORAGE_COUNT)
    storage_starts = q_starts + offs_b * PREFIX_PAD
    tile_count = (STORAGE_COUNT + BLOCK_N - 1) // BLOCK_N

    for tile_id in tl.range(pid, tile_count, n_programs):
        storage_row = tile_id * BLOCK_N + offs_n
        in_storage = storage_row < STORAGE_COUNT
        batch_id = (
            tl.sum(
                (
                    (storage_row[:, None] >= storage_starts[None, :]) & valid_b[None, :]
                ).to(tl.int32),
                axis=1,
            )
            - 1
        )
        valid_row = in_storage & (batch_id >= 0) & (batch_id < BATCH_SIZE)
        safe_batch = tl.maximum(batch_id, 0)
        q_start = tl.load(cu_seqlens_q_ptr + safe_batch, mask=valid_row, other=0)
        q_end = tl.load(cu_seqlens_q_ptr + safe_batch + 1, mask=valid_row, other=0)
        kv_len = tl.load(seqused_kv_ptr + safe_batch, mask=valid_row, other=0)
        local_storage = storage_row - (q_start + safe_batch * PREFIX_PAD)
        q_len = q_end - q_start
        logical_pos = kv_len - q_len - PREFIX_PAD + local_storage
        valid_row &= (
            (local_storage >= 0)
            & (local_storage < q_len + PREFIX_PAD)
            & (logical_pos >= 0)
        )
        tl.store(
            pack_plan_ptr + storage_row * 2,
            tl.where(valid_row, batch_id, -1),
            mask=in_storage,
        )
        tl.store(
            pack_plan_ptr + storage_row * 2 + 1,
            tl.where(valid_row, logical_pos, 0),
            mask=in_storage,
        )


@libentry()
@triton.jit(do_not_specialize=["STORAGE_COUNT"])
def _sparse_attn_sharedkv_eager_batched_pack_original_planned_kernel(
    logical_kv_ptr,
    logical_v_ptr,
    paged_kv_ptr,
    block_table_ptr,
    pack_plan_ptr,
    STORAGE_COUNT,
    TABLE_WIDTH: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Pack generic batched rows using a stream-cached host length plan."""
    tl.static_assert(HEAD_DIM == 512, "DSV4 planned PagePack expects D=512.")
    tl.static_assert(PAGE_SIZE == 128, "DSV4 planned PagePack expects page size 128.")
    tl.static_assert(BLOCK_N == 64, "Planned PagePack uses N64 tiles.")

    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    tiles_d: tl.constexpr = HEAD_DIM // BLOCK_D
    tile_count = (STORAGE_COUNT + BLOCK_N - 1) // BLOCK_N

    for task_id in tl.range(pid, tile_count * tiles_d, n_programs):
        tile_id = task_id // tiles_d
        d_tile = task_id - tile_id * tiles_d
        storage_row = tile_id * BLOCK_N + offs_n
        in_storage = storage_row < STORAGE_COUNT
        batch_id = tl.load(
            pack_plan_ptr + storage_row * 2,
            mask=in_storage,
            other=-1,
        )
        logical_pos = tl.load(
            pack_plan_ptr + storage_row * 2 + 1,
            mask=in_storage,
            other=0,
        )
        valid_row = in_storage & (batch_id >= 0)
        safe_batch = tl.maximum(batch_id, 0)
        logical_page = logical_pos // PAGE_SIZE
        page_offset = logical_pos - logical_page * PAGE_SIZE
        valid_row &= logical_page < TABLE_WIDTH
        physical_page = tl.load(
            block_table_ptr + safe_batch * TABLE_WIDTH + logical_page,
            mask=valid_row,
            other=0,
        )
        valid_row &= physical_page >= 0
        d_base = d_tile * BLOCK_D
        value = tl.load(
            paged_kv_ptr
            + (tl.maximum(physical_page, 0)[:, None] * PAGE_SIZE + page_offset[:, None])
            * HEAD_DIM
            + d_base
            + offs_d[None, :],
            mask=valid_row[:, None],
            other=0.0,
        )
        dst = storage_row[:, None] * HEAD_DIM + d_base + offs_d[None, :]
        tl.store(logical_kv_ptr + dst, value, mask=in_storage[:, None])
        tl.store(
            logical_v_ptr + dst,
            value.to(tl.float16),
            mask=in_storage[:, None],
        )


@libentry()
@triton.jit(do_not_specialize=["BATCH_SIZE"])
def _sparse_attn_sharedkv_eager_batched_pack_exact_page_kernel(
    logical_kv_ptr,
    logical_v_ptr,
    paged_kv_ptr,
    block_table_ptr,
    BATCH_SIZE,
    TABLE_WIDTH: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Copy exact q=kv=128 prompts without generic per-row page lookup."""
    tl.static_assert(HEAD_DIM == 512, "DSV4 exact PagePack expects D=512.")
    tl.static_assert(PAGE_SIZE == 128, "DSV4 exact PagePack expects one N128 page.")
    tl.static_assert(BLOCK_N == 128, "Exact PagePack uses a complete page per N tile.")

    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    tiles_d: tl.constexpr = HEAD_DIM // BLOCK_D
    task_count = BATCH_SIZE * tiles_d

    for task_id in tl.range(pid, task_count, n_programs):
        batch_id = task_id // tiles_d
        d_tile = task_id - batch_id * tiles_d
        physical_page = tl.load(block_table_ptr + batch_id * TABLE_WIDTH)
        d_base = d_tile * BLOCK_D
        value = tl.load(
            paged_kv_ptr
            + (tl.maximum(physical_page, 0) * PAGE_SIZE + offs_n[:, None]) * HEAD_DIM
            + d_base
            + offs_d[None, :],
            mask=physical_page >= 0,
            other=0.0,
        )
        dst = (
            (batch_id * PAGE_SIZE + offs_n[:, None]) * HEAD_DIM
            + d_base
            + offs_d[None, :]
        )
        tl.store(logical_kv_ptr + dst, value)
        tl.store(logical_v_ptr + dst, value.to(tl.float16))


@libentry()
@triton.jit(do_not_specialize=["BATCH_SIZE"])
def _sparse_attn_sharedkv_eager_batched_pack_exact_page_with_compressed_kernel(
    logical_ori_kv_ptr,
    logical_ori_v_ptr,
    logical_cmp_kv_ptr,
    logical_cmp_v_ptr,
    ori_paged_kv_ptr,
    cmp_paged_kv_ptr,
    ori_block_table_ptr,
    cmp_block_table_ptr,
    BATCH_SIZE,
    ORI_TABLE_WIDTH: tl.constexpr,
    CMP_TABLE_WIDTH: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    CMP_RATIO: tl.constexpr,
    CMP_STORAGE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CMP_BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Copy exact original and compressed pages in one model-path launch."""
    tl.static_assert(HEAD_DIM == 512, "DSV4 fused exact PagePack expects D=512.")
    tl.static_assert(PAGE_SIZE == 128, "DSV4 fused exact PagePack expects N128 pages.")
    tl.static_assert(BLOCK_N == 128, "Fused exact PagePack uses an original N128 tile.")
    tl.static_assert(
        CMP_BLOCK_N == 64, "Fused exact PagePack uses a compressed N64 tile."
    )
    tl.static_assert(
        CMP_RATIO == 4 or CMP_RATIO == 128, "Exact PagePack expects CFA or SCFA."
    )
    tl.static_assert(
        HEAD_DIM % BLOCK_D == 0, "Fused exact PagePack requires complete D tiles."
    )
    tl.static_assert(
        CMP_STORAGE == CMP_BLOCK_N,
        "Exact compressed PagePack uses one complete N64 buffer.",
    )

    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)
    offs_n = tl.arange(0, BLOCK_N)
    offs_cmp_n = tl.arange(0, CMP_BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    tiles_d: tl.constexpr = HEAD_DIM // BLOCK_D
    task_count = BATCH_SIZE * tiles_d

    for task_id in tl.range(pid, task_count, n_programs):
        batch_id = task_id // tiles_d
        d_tile = task_id - batch_id * tiles_d
        d_base = d_tile * BLOCK_D

        ori_page = tl.load(ori_block_table_ptr + batch_id * ORI_TABLE_WIDTH)
        ori_value = tl.load(
            ori_paged_kv_ptr
            + (tl.maximum(ori_page, 0) * PAGE_SIZE + offs_n[:, None]) * HEAD_DIM
            + d_base
            + offs_d[None, :],
            mask=ori_page >= 0,
            other=0.0,
        )
        ori_dst = (
            (batch_id * PAGE_SIZE + offs_n[:, None]) * HEAD_DIM
            + d_base
            + offs_d[None, :]
        )
        tl.store(logical_ori_kv_ptr + ori_dst, ori_value)
        tl.store(logical_ori_v_ptr + ori_dst, ori_value.to(tl.float16))

        cmp_page = tl.load(cmp_block_table_ptr + batch_id * CMP_TABLE_WIDTH)
        cmp_valid = offs_cmp_n < (PAGE_SIZE // CMP_RATIO)
        cmp_value = tl.load(
            cmp_paged_kv_ptr
            + (tl.maximum(cmp_page, 0) * PAGE_SIZE + offs_cmp_n[:, None]) * HEAD_DIM
            + d_base
            + offs_d[None, :],
            mask=(cmp_page >= 0) & cmp_valid[:, None],
            other=0.0,
        )
        cmp_dst = (
            (batch_id * CMP_STORAGE + offs_cmp_n[:, None]) * HEAD_DIM
            + d_base
            + offs_d[None, :]
        )
        tl.store(logical_cmp_kv_ptr + cmp_dst, cmp_value)
        tl.store(logical_cmp_v_ptr + cmp_dst, cmp_value.to(tl.float16))


@libentry()
@triton.jit(do_not_specialize=["BATCH_SIZE"])
def _sparse_attn_sharedkv_eager_batched_pack_compressed_kernel(
    logical_kv_ptr,
    logical_v_ptr,
    paged_kv_ptr,
    block_table_ptr,
    seqused_kv_ptr,
    BATCH_SIZE,
    STORAGE_STRIDE: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    CMP_RATIO: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """按固定 batch stride 一次整理所有序列的 compressed 前缀。"""
    tl.static_assert(HEAD_DIM == 512, "DSV4 batched compressed pack expects D=512.")
    tl.static_assert(BLOCK_N == 64, "Batched compressed pack uses N64 tiles.")

    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    tiles_d: tl.constexpr = HEAD_DIM // BLOCK_D
    tiles_per_batch: tl.constexpr = (STORAGE_STRIDE + BLOCK_N - 1) // BLOCK_N
    task_count = BATCH_SIZE * tiles_per_batch * tiles_d

    for task_id in tl.range(pid, task_count, n_programs):
        batch_id = task_id // (tiles_per_batch * tiles_d)
        local_task = task_id - batch_id * tiles_per_batch * tiles_d
        tile_id = local_task // tiles_d
        d_tile = local_task - tile_id * tiles_d
        logical_pos = tile_id * BLOCK_N + offs_n
        kv_len = tl.load(seqused_kv_ptr + batch_id)
        valid = logical_pos < (kv_len // CMP_RATIO)
        logical_page = logical_pos // PAGE_SIZE
        page_offset = logical_pos - logical_page * PAGE_SIZE
        valid &= logical_page < TABLE_WIDTH
        physical_page = tl.load(
            block_table_ptr + batch_id * TABLE_WIDTH + logical_page,
            mask=valid,
            other=0,
        )
        valid &= physical_page >= 0
        d_base = d_tile * BLOCK_D
        value = tl.load(
            paged_kv_ptr
            + (tl.maximum(physical_page, 0)[:, None] * PAGE_SIZE + page_offset[:, None])
            * HEAD_DIM
            + d_base
            + offs_d[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        dst_row = batch_id * STORAGE_STRIDE + logical_pos
        tl.store(
            logical_kv_ptr + dst_row[:, None] * HEAD_DIM + d_base + offs_d[None, :],
            value,
            mask=(logical_pos < STORAGE_STRIDE)[:, None],
        )
        tl.store(
            logical_v_ptr + dst_row[:, None] * HEAD_DIM + d_base + offs_d[None, :],
            value.to(tl.float16),
            mask=(logical_pos < STORAGE_STRIDE)[:, None],
        )


@libentry()
@triton.jit(do_not_specialize=["TOTAL_Q"])
def _sparse_attn_sharedkv_eager_batched_staged_qk_softmax_kernel(
    ori_prob_ptr,
    cmp_prob_ptr,
    q_ptr,
    logical_ori_k_ptr,
    logical_cmp_k_ptr,
    q_plan_ptr,
    sinks_ptr,
    softmax_scale,
    TOTAL_Q,
    NUM_Q_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PREFIX_PAD: tl.constexpr,
    ORI_N: tl.constexpr,
    CMP_STORAGE: tl.constexpr,
    CMP_RATIO: tl.constexpr,
    HAS_CMP: tl.constexpr,
    D_BLOCK: tl.constexpr,
):
    """多batch staged MM1+softmax；一个task覆盖一个token的全部64个head。"""
    tl.static_assert(NUM_Q_HEADS == 64, "Batched staged QK expects 64 shared-KV heads.")
    tl.static_assert(HEAD_DIM == 512, "Batched staged QK expects D=512.")
    tl.static_assert(ORI_N == 128, "Batched staged QK expects an N128 original window.")
    tl.static_assert(
        not HAS_CMP or CMP_STORAGE == 64, "The compact staged branch uses one N64 tile."
    )

    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)
    offs_h = tl.arange(0, NUM_Q_HEADS)
    offs_ori_n = tl.arange(0, ORI_N)
    offs_cmp_n = tl.arange(0, CMP_STORAGE)
    offs_d = tl.arange(0, D_BLOCK)
    sink = tl.load(sinks_ptr + offs_h).to(tl.float32)

    for q_row in tl.range(pid, TOTAL_Q, n_programs):
        safe_batch = tl.load(q_plan_ptr + q_row * 2)
        q_position = tl.load(q_plan_ptr + q_row * 2 + 1)
        if PREFIX_PAD == 0:
            ori_storage_start = q_row - q_position
            valid_ori = offs_ori_n <= q_position
        else:
            ori_storage_start = q_row + safe_batch * PREFIX_PAD
            valid_ori = offs_ori_n >= tl.maximum(PREFIX_PAD - q_position, 0)
        ori_scores = tl.zeros((NUM_Q_HEADS, ORI_N), tl.float32)
        for d_base in tl.static_range(0, HEAD_DIM, D_BLOCK):
            q = tl.load(
                q_ptr
                + (q_row * NUM_Q_HEADS + offs_h[:, None]) * HEAD_DIM
                + d_base
                + offs_d[None, :]
            )
            ori_k = tl.load(
                logical_ori_k_ptr
                + (ori_storage_start + offs_ori_n[:, None]) * HEAD_DIM
                + d_base
                + offs_d[None, :]
            )
            ori_scores = tl.dot(q, tl.trans(ori_k), acc=ori_scores)
        ori_scores *= softmax_scale
        ori_scores = tl.where(valid_ori[None, :], ori_scores, -1e6)

        row_max = tl.maximum(sink, tl.max(ori_scores, axis=1))
        if HAS_CMP:
            cmp_threshold = (q_position + 1) // CMP_RATIO
            cmp_scores = tl.zeros((NUM_Q_HEADS, CMP_STORAGE), tl.float32)
            valid_cmp = offs_cmp_n < cmp_threshold
            for d_base in tl.static_range(0, HEAD_DIM, D_BLOCK):
                q = tl.load(
                    q_ptr
                    + (q_row * NUM_Q_HEADS + offs_h[:, None]) * HEAD_DIM
                    + d_base
                    + offs_d[None, :]
                )
                cmp_k = tl.load(
                    logical_cmp_k_ptr
                    + (safe_batch * CMP_STORAGE + offs_cmp_n[:, None]) * HEAD_DIM
                    + d_base
                    + offs_d[None, :]
                )
                cmp_scores = tl.dot(q, tl.trans(cmp_k), acc=cmp_scores)
            cmp_scores *= softmax_scale
            cmp_scores = tl.where(valid_cmp[None, :], cmp_scores, -1e6)
            row_max = tl.maximum(row_max, tl.max(cmp_scores, axis=1))

        ori_prob = tl.exp(ori_scores - row_max[:, None])
        ori_prob = tl.where(valid_ori[None, :], ori_prob, 0.0)
        denom = tl.exp(sink - row_max) + tl.sum(ori_prob, axis=1)
        if HAS_CMP:
            cmp_prob = tl.exp(cmp_scores - row_max[:, None])
            cmp_prob = tl.where(valid_cmp[None, :], cmp_prob, 0.0)
            denom += tl.sum(cmp_prob, axis=1)
        inv_denom = cann_libdevice.reciprocal(denom)
        ori_prob *= inv_denom[:, None]
        tl.store(
            ori_prob_ptr
            + (q_row * NUM_Q_HEADS + offs_h[:, None]) * ORI_N
            + offs_ori_n[None, :],
            ori_prob,
        )
        if HAS_CMP:
            cmp_prob *= inv_denom[:, None]
            tl.store(
                cmp_prob_ptr
                + (q_row * NUM_Q_HEADS + offs_h[:, None]) * CMP_STORAGE
                + offs_cmp_n[None, :],
                cmp_prob,
            )


@libentry()
@triton.jit(do_not_specialize=["TOTAL_Q"])
def _sparse_attn_sharedkv_eager_batched_scfa_original_qk_kernel(
    ori_prob_ptr,
    ori_max_ptr,
    ori_sum_ptr,
    q_ptr,
    logical_ori_k_ptr,
    q_plan_ptr,
    softmax_scale,
    TOTAL_Q,
    NUM_Q_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PREFIX_PAD: tl.constexpr,
    ORI_N: tl.constexpr,
    D_BLOCK: tl.constexpr,
):
    """生成batched SCFA original分支的未归一化(m,l,P)状态。"""
    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)
    offs_h = tl.arange(0, NUM_Q_HEADS)
    offs_n = tl.arange(0, ORI_N)
    offs_d = tl.arange(0, D_BLOCK)

    for q_row in tl.range(pid, TOTAL_Q, n_programs):
        safe_batch = tl.load(q_plan_ptr + q_row * 2)
        q_position = tl.load(q_plan_ptr + q_row * 2 + 1)
        if PREFIX_PAD == 0:
            storage_start = q_row - q_position
            valid_n = offs_n <= q_position
        else:
            storage_start = q_row + safe_batch * PREFIX_PAD
            valid_n = offs_n >= tl.maximum(PREFIX_PAD - q_position, 0)
        scores = tl.zeros((NUM_Q_HEADS, ORI_N), tl.float32)
        for d_base in tl.static_range(0, HEAD_DIM, D_BLOCK):
            q = tl.load(
                q_ptr
                + (q_row * NUM_Q_HEADS + offs_h[:, None]) * HEAD_DIM
                + d_base
                + offs_d[None, :]
            )
            k = tl.load(
                logical_ori_k_ptr
                + (storage_start + offs_n[:, None]) * HEAD_DIM
                + d_base
                + offs_d[None, :]
            )
            scores = tl.dot(q, tl.trans(k), acc=scores)
        scores *= softmax_scale
        scores = tl.where(valid_n[None, :], scores, -1e6)
        row_max = tl.max(scores, axis=1)
        prob = tl.exp(scores - row_max[:, None])
        prob = tl.where(valid_n[None, :], prob, 0.0)
        row_sum = tl.sum(prob, axis=1)
        tl.store(
            ori_prob_ptr
            + (q_row * NUM_Q_HEADS + offs_h[:, None]) * ORI_N
            + offs_n[None, :],
            prob,
        )
        tl.store(ori_max_ptr + q_row * NUM_Q_HEADS + offs_h, row_max)
        tl.store(ori_sum_ptr + q_row * NUM_Q_HEADS + offs_h, row_sum)


@libentry()
@triton.jit(do_not_specialize=["TOTAL_Q"])
def _sparse_attn_sharedkv_eager_batched_staged_pv_kernel(
    out_ptr,
    ori_prob_ptr,
    cmp_prob_ptr,
    logical_ori_v_ptr,
    logical_cmp_v_ptr,
    q_plan_ptr,
    TOTAL_Q,
    NUM_Q_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PREFIX_PAD: tl.constexpr,
    ORI_N: tl.constexpr,
    CMP_STORAGE: tl.constexpr,
    HAS_CMP: tl.constexpr,
):
    """多batch staged MM2；P和FP16 V相乘后直接写最终O。"""
    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)
    offs_h = tl.arange(0, NUM_Q_HEADS)
    offs_ori_n = tl.arange(0, ORI_N)
    offs_cmp_n = tl.arange(0, CMP_STORAGE)
    offs_d = tl.arange(0, HEAD_DIM)

    for q_row in tl.range(pid, TOTAL_Q, n_programs, loop_unroll_factor=2):
        safe_batch = tl.load(q_plan_ptr + q_row * 2)
        q_position = tl.load(q_plan_ptr + q_row * 2 + 1)
        if PREFIX_PAD == 0:
            ori_storage_start = q_row - q_position
        else:
            ori_storage_start = q_row + safe_batch * PREFIX_PAD
        ori_prob = tl.load(
            ori_prob_ptr
            + (q_row * NUM_Q_HEADS + offs_h[:, None]) * ORI_N
            + offs_ori_n[None, :]
        )
        ori_v = tl.load(
            logical_ori_v_ptr
            + (ori_storage_start + offs_ori_n[:, None]) * HEAD_DIM
            + offs_d[None, :]
        )
        acc = tl.dot(ori_prob, ori_v)
        if HAS_CMP:
            cmp_prob = tl.load(
                cmp_prob_ptr
                + (q_row * NUM_Q_HEADS + offs_h[:, None]) * CMP_STORAGE
                + offs_cmp_n[None, :]
            )
            cmp_v = tl.load(
                logical_cmp_v_ptr
                + (safe_batch * CMP_STORAGE + offs_cmp_n[:, None]) * HEAD_DIM
                + offs_d[None, :]
            )
            acc = tl.dot(cmp_prob, cmp_v, acc=acc)
        tl.store(
            out_ptr
            + (q_row * NUM_Q_HEADS + offs_h[:, None]) * HEAD_DIM
            + offs_d[None, :],
            acc,
        )


@libentry()
@triton.jit(do_not_specialize=["TOTAL_Q", "BATCH_SIZE"])
def _sparse_attn_sharedkv_decode_dynamic_pack_original_n64_kernel(
    ori_kv_ptr,
    ori_block_table_ptr,
    cu_seqlens_q_ptr,
    seqused_kv_ptr,
    compact_kv_ptr,
    compact_valid_ptr,
    TOTAL_Q,
    BATCH_SIZE,
    BATCH_BLOCK: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ORI_BLOCK_SIZE: tl.constexpr,
    ORI_TABLE_WIDTH: tl.constexpr,
    MAX_COMPACT_TOKENS: tl.constexpr,
    TILES_PER_Q: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """无循环地把每个 original tile 整理到 compact workspace。"""
    tl.static_assert(BLOCK_N == 64, "Graph-safe original pack expects N64.")
    tl.static_assert(BLOCK_D == 512, "Graph-safe original pack expects D512.")
    pid = tl.program_id(0)
    q_row = pid // TILES_PER_Q
    tile_id = pid - q_row * TILES_PER_Q
    valid_row, ori_threshold, safe_batch = _graphsafe_q_row_info_i32(
        q_row,
        cu_seqlens_q_ptr,
        seqused_kv_ptr,
        BATCH_SIZE,
        BATCH_BLOCK,
    )
    slot = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)
    ori_pos = ori_threshold - 128 + slot
    valid_ori = valid_row & (ori_pos >= 0) & (ori_pos < ori_threshold)
    safe_ori_pos = tl.maximum(ori_pos, 0)
    ori_page = safe_ori_pos // ORI_BLOCK_SIZE
    ori_offset = safe_ori_pos - ori_page * ORI_BLOCK_SIZE
    valid_ori &= ori_page < ORI_TABLE_WIDTH
    physical_page = tl.load(
        ori_block_table_ptr + safe_batch * ORI_TABLE_WIDTH + ori_page,
        mask=valid_ori,
        other=0,
    )
    valid_ori &= physical_page >= 0
    offs_d = tl.arange(0, BLOCK_D)
    compact_base = (q_row * MAX_COMPACT_TOKENS + slot) * HEAD_DIM
    if valid_row:
        for d_base in tl.static_range(0, HEAD_DIM, BLOCK_D):
            src = tl.load(
                ori_kv_ptr
                + (
                    tl.maximum(physical_page, 0)[:, None] * ORI_BLOCK_SIZE
                    + ori_offset[:, None]
                )
                * HEAD_DIM
                + d_base
                + offs_d[None, :],
            )
            tl.store(
                compact_kv_ptr + compact_base[:, None] + d_base + offs_d[None, :],
                src,
            )
    tl.store(compact_valid_ptr + q_row * MAX_COMPACT_TOKENS + slot, valid_ori)


@libentry()
@triton.jit(do_not_specialize=["TOTAL_Q", "BATCH_SIZE"])
def _sparse_attn_sharedkv_decode_dynamic_pack_compressed_n64_kernel(
    cmp_kv_ptr,
    cmp_sparse_indices_ptr,
    cmp_block_table_ptr,
    cu_seqlens_q_ptr,
    seqused_kv_ptr,
    compact_kv_ptr,
    compact_valid_ptr,
    TOTAL_Q,
    BATCH_SIZE,
    BATCH_BLOCK: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CMP_BLOCK_SIZE: tl.constexpr,
    CMP_TABLE_WIDTH: tl.constexpr,
    CMP_RATIO: tl.constexpr,
    MODE: tl.constexpr,
    MAX_CMP_TOKENS: tl.constexpr,
    MAX_COMPACT_TOKENS: tl.constexpr,
    TILES_PER_Q: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """每个 program 搬一个 N64 compressed tile，不保留运行时循环。"""
    tl.static_assert(BLOCK_N == 64, "Graph-safe compressed pack expects N64.")
    tl.static_assert(BLOCK_D == 512, "Graph-safe compressed pack expects D512.")
    tl.static_assert(MODE == 1 or MODE == 2, "Compressed pack supports CFA/SCFA.")
    pid = tl.program_id(0)
    q_row = pid // TILES_PER_Q
    tile_id = pid - q_row * TILES_PER_Q
    valid_row, ori_threshold, safe_batch = _graphsafe_q_row_info_i32(
        q_row,
        cu_seqlens_q_ptr,
        seqused_kv_ptr,
        BATCH_SIZE,
        BATCH_BLOCK,
    )
    cmp_threshold = ori_threshold // CMP_RATIO
    if MODE == 1:
        cmp_count = tl.minimum(cmp_threshold, MAX_CMP_TOKENS)
    else:
        cmp_count = tl.minimum(cmp_threshold, 512)
    cmp_slot = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)
    capacity_slot = cmp_slot < MAX_CMP_TOKENS
    valid_slot = valid_row & capacity_slot & (cmp_slot < cmp_count)
    if MODE == 1:
        logical_cmp_pos = cmp_slot
        valid_cmp = valid_slot
    else:
        sparse_idx = tl.load(
            cmp_sparse_indices_ptr + q_row * 512 + cmp_slot,
            mask=valid_slot,
            other=-1,
        )
        logical_cmp_pos = tl.maximum(sparse_idx, 0)
        valid_cmp = valid_slot & (sparse_idx >= 0) & (sparse_idx < cmp_threshold)

    cmp_page = logical_cmp_pos // CMP_BLOCK_SIZE
    cmp_offset = logical_cmp_pos - cmp_page * CMP_BLOCK_SIZE
    valid_cmp &= cmp_page < CMP_TABLE_WIDTH
    physical_page = tl.load(
        cmp_block_table_ptr + safe_batch * CMP_TABLE_WIDTH + cmp_page,
        mask=valid_cmp,
        other=0,
    )
    valid_cmp &= physical_page >= 0
    compact_slot = 128 + cmp_slot
    compact_base = (q_row * MAX_COMPACT_TOKENS + compact_slot) * HEAD_DIM
    offs_d = tl.arange(0, BLOCK_D)
    has_valid_token = tl.sum(valid_cmp.to(tl.int32), axis=0) > 0
    if has_valid_token:
        for d_base in tl.static_range(0, HEAD_DIM, BLOCK_D):
            src = tl.load(
                cmp_kv_ptr
                + (
                    tl.maximum(physical_page, 0)[:, None] * CMP_BLOCK_SIZE
                    + cmp_offset[:, None]
                )
                * HEAD_DIM
                + d_base
                + offs_d[None, :],
            )
            tl.store(
                compact_kv_ptr + compact_base[:, None] + d_base + offs_d[None, :],
                src,
                mask=capacity_slot[:, None],
            )
    tl.store(
        compact_valid_ptr + q_row * MAX_COMPACT_TOKENS + compact_slot,
        valid_cmp,
        mask=capacity_slot,
    )


@libentry()
@triton.jit(do_not_specialize=["BATCH_SIZE"])
def _sparse_attn_sharedkv_q1_scfa_pack_kernel(
    ori_kv_ptr,
    cmp_kv_ptr,
    sparse_indices_ptr,
    ori_table_ptr,
    cmp_table_ptr,
    cu_q_ptr,
    used_kv_ptr,
    compact_ptr,
    valid_ptr,
    BATCH_SIZE,
    BATCH_BLOCK: tl.constexpr,
    ORI_TABLE_WIDTH: tl.constexpr,
    CMP_TABLE_WIDTH: tl.constexpr,
):
    """Pack two original and eight sparse tiles for one q1 row in one launch."""
    tile = tl.program_id(0)
    valid_row, threshold, batch = _graphsafe_q_row_info_i32(
        0, cu_q_ptr, used_kv_ptr, BATCH_SIZE, BATCH_BLOCK
    )
    n = tl.arange(0, 64)
    d = tl.arange(0, 512)
    slot = tile * 64 + n
    if tile < 2:
        position = threshold - 128 + slot
        valid = valid_row & (position >= 0) & (position < threshold)
        safe_position = tl.maximum(position, 0)
        page = safe_position // 128
        offset = safe_position % 128
        valid &= page < ORI_TABLE_WIDTH
        physical = tl.load(
            ori_table_ptr + batch * ORI_TABLE_WIDTH + page, mask=valid, other=0
        )
        valid &= physical >= 0
        if valid_row:
            values = tl.load(
                ori_kv_ptr
                + (tl.maximum(physical, 0)[:, None] * 128 + offset[:, None]) * 512
                + d[None, :]
            )
            tl.store(compact_ptr + slot[:, None] * 512 + d[None, :], values)
    else:
        sparse_slot = slot - 128
        cmp_threshold = threshold // 4
        valid_slot = valid_row & (sparse_slot < tl.minimum(cmp_threshold, 512))
        position = tl.load(sparse_indices_ptr + sparse_slot, mask=valid_slot, other=-1)
        valid = valid_slot & (position >= 0) & (position < cmp_threshold)
        safe_position = tl.maximum(position, 0)
        page = safe_position // 128
        offset = safe_position % 128
        valid &= page < CMP_TABLE_WIDTH
        physical = tl.load(
            cmp_table_ptr + batch * CMP_TABLE_WIDTH + page, mask=valid, other=0
        )
        valid &= physical >= 0
        if tl.sum(valid.to(tl.int32), axis=0) > 0:
            values = tl.load(
                cmp_kv_ptr
                + (tl.maximum(physical, 0)[:, None] * 128 + offset[:, None]) * 512
                + d[None, :]
            )
            tl.store(compact_ptr + slot[:, None] * 512 + d[None, :], values)
    tl.store(valid_ptr + slot, valid)


@libentry()
@triton.jit(do_not_specialize=["BATCH_SIZE"])
def _sparse_attn_sharedkv_q8_scfa_pack_kernel(
    ori_kv_ptr,
    cmp_kv_ptr,
    sparse_indices_ptr,
    ori_table_ptr,
    cmp_table_ptr,
    cu_q_ptr,
    used_kv_ptr,
    compact_ptr,
    valid_ptr,
    BATCH_SIZE,
    BATCH_BLOCK: tl.constexpr,
    ORI_TABLE_WIDTH: tl.constexpr,
    CMP_TABLE_WIDTH: tl.constexpr,
):
    """Pack two original and eight sparse tiles per q row in one launch."""
    pid = tl.program_id(0)
    q_row = pid // 10
    tile = pid - q_row * 10
    valid_row, threshold, batch = _graphsafe_q_row_info_i32(
        q_row, cu_q_ptr, used_kv_ptr, BATCH_SIZE, BATCH_BLOCK
    )
    n = tl.arange(0, 64)
    d = tl.arange(0, 512)
    slot = tile * 64 + n
    compact_slot = q_row * 640 + slot
    if tile < 2:
        position = threshold - 128 + slot
        valid = valid_row & (position >= 0) & (position < threshold)
        safe_position = tl.maximum(position, 0)
        page = safe_position // 128
        offset = safe_position % 128
        valid &= page < ORI_TABLE_WIDTH
        physical = tl.load(
            ori_table_ptr + batch * ORI_TABLE_WIDTH + page, mask=valid, other=0
        )
        valid &= physical >= 0
        if valid_row:
            values = tl.load(
                ori_kv_ptr
                + (tl.maximum(physical, 0)[:, None] * 128 + offset[:, None]) * 512
                + d[None, :]
            )
            tl.store(compact_ptr + compact_slot[:, None] * 512 + d[None, :], values)
    else:
        sparse_slot = slot - 128
        cmp_threshold = threshold // 4
        valid_slot = valid_row & (sparse_slot < tl.minimum(cmp_threshold, 512))
        position = tl.load(
            sparse_indices_ptr + q_row * 512 + sparse_slot,
            mask=valid_slot,
            other=-1,
        )
        valid = valid_slot & (position >= 0) & (position < cmp_threshold)
        safe_position = tl.maximum(position, 0)
        page = safe_position // 128
        offset = safe_position % 128
        valid &= page < CMP_TABLE_WIDTH
        physical = tl.load(
            cmp_table_ptr + batch * CMP_TABLE_WIDTH + page, mask=valid, other=0
        )
        valid &= physical >= 0
        if tl.sum(valid.to(tl.int32), axis=0) > 0:
            values = tl.load(
                cmp_kv_ptr
                + (tl.maximum(physical, 0)[:, None] * 128 + offset[:, None]) * 512
                + d[None, :]
            )
            tl.store(compact_ptr + compact_slot[:, None] * 512 + d[None, :], values)
    tl.store(valid_ptr + compact_slot, valid)


@libentry()
@triton.jit(do_not_specialize=["TOTAL_Q", "BATCH_SIZE"])
def _sparse_attn_sharedkv_q1_q2_cfa_fused_pack_kernel(
    ori_kv_ptr,
    cmp_kv_ptr,
    ori_block_table_ptr,
    cmp_block_table_ptr,
    cu_seqlens_q_ptr,
    seqused_kv_ptr,
    compact_kv_ptr,
    compact_valid_ptr,
    TOTAL_Q,
    BATCH_SIZE,
    BATCH_BLOCK: tl.constexpr,
    ORI_TABLE_WIDTH: tl.constexpr,
    CMP_TABLE_WIDTH: tl.constexpr,
    MAX_CMP_TOKENS: tl.constexpr,
    MAX_COMPACT_TOKENS: tl.constexpr,
):
    """Two original tiles and one dynamic compressed task per q, in one launch."""
    pid = tl.program_id(0)
    q_row = pid // 3
    task = pid - q_row * 3
    valid_row, threshold, batch = _graphsafe_q_row_info_i32(
        q_row, cu_seqlens_q_ptr, seqused_kv_ptr, BATCH_SIZE, BATCH_BLOCK
    )
    n = tl.arange(0, 64)
    d = tl.arange(0, 512)
    if task < 2:
        slot = task * 64 + n
        position = threshold - 128 + slot
        valid = valid_row & (position >= 0) & (position < threshold)
        safe_position = tl.maximum(position, 0)
        page = safe_position // 128
        offset = safe_position % 128
        valid &= page < ORI_TABLE_WIDTH
        physical = tl.load(
            ori_block_table_ptr + batch * ORI_TABLE_WIDTH + page,
            mask=valid,
            other=0,
        )
        valid &= physical >= 0
        if valid_row:
            values = tl.load(
                ori_kv_ptr
                + (tl.maximum(physical, 0)[:, None] * 128 + offset[:, None]) * 512
                + d[None, :]
            )
            tl.store(
                compact_kv_ptr
                + (q_row * MAX_COMPACT_TOKENS + slot[:, None]) * 512
                + d[None, :],
                values,
            )
        tl.store(compact_valid_ptr + q_row * MAX_COMPACT_TOKENS + slot, valid)
    elif valid_row:
        cmp_count = tl.minimum(threshold // 128, MAX_CMP_TOKENS)
        for base in tl.range(0, cmp_count, 64):
            position = base + n
            valid = position < cmp_count
            safe_position = tl.minimum(position, tl.maximum(cmp_count - 1, 0))
            page = safe_position // 128
            offset = safe_position % 128
            valid &= page < CMP_TABLE_WIDTH
            physical = tl.load(
                cmp_block_table_ptr + batch * CMP_TABLE_WIDTH + page,
                mask=valid,
                other=0,
            )
            valid &= physical >= 0
            values = tl.load(
                cmp_kv_ptr
                + (tl.maximum(physical, 0)[:, None] * 128 + offset[:, None]) * 512
                + d[None, :]
            )
            slot = 128 + position
            tl.store(
                compact_kv_ptr
                + (q_row * MAX_COMPACT_TOKENS + slot[:, None]) * 512
                + d[None, :],
                values,
            )
            tl.store(compact_valid_ptr + q_row * MAX_COMPACT_TOKENS + slot, valid)


@libentry()
@triton.jit(do_not_specialize=["TOTAL_Q", "BATCH_SIZE"])
def _sparse_attn_sharedkv_decode_dynamic_pack_cfa_rows_kernel(
    cmp_kv_ptr,
    cmp_block_table_ptr,
    cu_seqlens_q_ptr,
    seqused_kv_ptr,
    compact_kv_ptr,
    compact_valid_ptr,
    TOTAL_Q,
    BATCH_SIZE,
    BATCH_BLOCK: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CMP_BLOCK_SIZE: tl.constexpr,
    CMP_TABLE_WIDTH: tl.constexpr,
    CMP_RATIO: tl.constexpr,
    MAX_CMP_TOKENS: tl.constexpr,
    MAX_COMPACT_TOKENS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """CFA按实际compressed长度pack，避免静态最大容量产生空program。"""
    tl.static_assert(BLOCK_N == 64, "Graph-safe CFA row pack expects N64.")
    tl.static_assert(BLOCK_D == 512, "Graph-safe CFA row pack expects D512.")

    q_row = tl.program_id(0)
    valid_row, ori_threshold, safe_batch = _graphsafe_q_row_info_i32(
        q_row,
        cu_seqlens_q_ptr,
        seqused_kv_ptr,
        BATCH_SIZE,
        BATCH_BLOCK,
    )
    cmp_count = tl.minimum(ori_threshold // CMP_RATIO, MAX_CMP_TOKENS)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    if valid_row:
        for cmp_base in tl.range(0, cmp_count, BLOCK_N):
            cmp_pos = cmp_base + offs_n
            valid_cmp = cmp_pos < cmp_count
            safe_cmp_pos = tl.minimum(cmp_pos, tl.maximum(cmp_count - 1, 0))
            cmp_page = safe_cmp_pos // CMP_BLOCK_SIZE
            cmp_offset = safe_cmp_pos - cmp_page * CMP_BLOCK_SIZE
            valid_cmp &= cmp_page < CMP_TABLE_WIDTH
            physical_page = tl.load(
                cmp_block_table_ptr + safe_batch * CMP_TABLE_WIDTH + cmp_page,
                mask=valid_cmp,
                other=0,
            )
            valid_cmp &= physical_page >= 0
            src = tl.load(
                cmp_kv_ptr
                + (
                    tl.maximum(physical_page, 0)[:, None] * CMP_BLOCK_SIZE
                    + cmp_offset[:, None]
                )
                * HEAD_DIM
                + offs_d[None, :]
            )
            compact_slot = 128 + cmp_pos
            compact_base = (q_row * MAX_COMPACT_TOKENS + compact_slot) * HEAD_DIM
            tl.store(
                compact_kv_ptr + compact_base[:, None] + offs_d[None, :],
                src,
            )
            tl.store(
                compact_valid_ptr + q_row * MAX_COMPACT_TOKENS + compact_slot,
                valid_cmp,
            )


@libentry()
@triton.jit(do_not_specialize=["TOTAL_Q", "BATCH_SIZE"])
def _sparse_attn_sharedkv_decode_dynamic_compact_full_attention_kernel(
    q_ptr,
    compact_kv_ptr,
    compact_valid_ptr,
    cu_seqlens_q_ptr,
    seqused_kv_ptr,
    sinks_ptr,
    out_ptr,
    softmax_scale,
    TOTAL_Q,
    BATCH_SIZE,
    BATCH_BLOCK: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    MAX_COMPACT_TOKENS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    MODE: tl.constexpr,
    STATIC_SWA_LOOP: tl.constexpr,
):
    """SWA/CFA 全有效且 N64 对齐时的无 token-mask attention。"""
    tl.static_assert(HEAD_DIM == 512, "DSV4 compact attention expects D=512.")
    tl.static_assert(BLOCK_N == 64, "DSV4 full compact attention expects N64.")
    tl.static_assert(MODE == 0 or MODE == 1, "Full compact path only supports SWA/CFA.")

    pid = tl.program_id(0)
    head_blocks = tl.cdiv(NUM_Q_HEADS, HEAD_BLOCK)
    q_row = (pid // head_blocks).to(tl.int64)
    head_block_id = pid - q_row * head_blocks
    valid_row, ori_threshold, _safe_batch = _graphsafe_q_row_threshold(
        q_row,
        cu_seqlens_q_ptr,
        seqused_kv_ptr,
        BATCH_SIZE,
        BATCH_BLOCK,
    )
    all_valid = ori_threshold >= 128
    if MODE == 0:
        compact_count = 128
        fast_row = valid_row
    else:
        cmp_count = tl.minimum(ori_threshold // 128, MAX_COMPACT_TOKENS - 128)
        compact_count = 128 + cmp_count
        fast_row = valid_row & all_valid & (compact_count % BLOCK_N == 0)

    offs_h = head_block_id.to(tl.int64) * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK).to(
        tl.int64
    )
    valid_h = offs_h < NUM_Q_HEADS
    offs_d = tl.arange(0, HEAD_DIM)
    if fast_row:
        offs_n = tl.arange(0, BLOCK_N)
        q = tl.load(
            q_ptr
            + (q_row * NUM_Q_HEADS + offs_h[:, None]) * HEAD_DIM
            + offs_d[None, :],
            mask=valid_h[:, None],
            other=0.0,
        )
        row_max = tl.load(sinks_ptr + offs_h, mask=valid_h, other=0.0).to(tl.float32)
        row_sum = tl.where(valid_h, 1.0, 0.0).to(tl.float32)
        acc = tl.zeros((HEAD_BLOCK, HEAD_DIM), dtype=tl.float32)
        if MODE == 0 and STATIC_SWA_LOOP:
            for token_base in tl.static_range(0, 128, BLOCK_N):
                token = token_base + offs_n.to(tl.int64)
                valid_token = tl.load(
                    compact_valid_ptr + q_row * MAX_COMPACT_TOKENS + token
                ).to(tl.int1)
                kv = tl.load(
                    compact_kv_ptr
                    + ((q_row * MAX_COMPACT_TOKENS + token)[:, None] * HEAD_DIM)
                    + offs_d[None, :]
                )
                scores = tl.dot(q, tl.trans(kv)) * softmax_scale
                scores = tl.where(
                    valid_h[:, None] & valid_token[None, :],
                    scores,
                    float("-inf"),
                )
                new_max = tl.maximum(row_max, tl.max(scores, axis=1))
                p = tl.exp(scores - new_max[:, None])
                p = tl.where(valid_h[:, None] & valid_token[None, :], p, 0.0)
                alpha = tl.exp(row_max - new_max)
                row_sum = row_sum * alpha + tl.sum(p, axis=1)
                acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
                row_max = new_max
        else:
            for token_base in tl.range(0, compact_count, BLOCK_N):
                token = token_base + offs_n.to(tl.int64)
                kv = tl.load(
                    compact_kv_ptr
                    + ((q_row * MAX_COMPACT_TOKENS + token)[:, None] * HEAD_DIM)
                    + offs_d[None, :]
                )
                scores = tl.dot(q, tl.trans(kv)) * softmax_scale
                scores = tl.where(valid_h[:, None], scores, float("-inf"))
                new_max = tl.maximum(row_max, tl.max(scores, axis=1))
                p = tl.exp(scores - new_max[:, None])
                p = tl.where(valid_h[:, None], p, 0.0)
                alpha = tl.exp(row_max - new_max)
                row_sum = row_sum * alpha + tl.sum(p, axis=1)
                acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
                row_max = new_max

        tl.store(
            out_ptr
            + (q_row * NUM_Q_HEADS + offs_h[:, None]) * HEAD_DIM
            + offs_d[None, :],
            (acc / row_sum[:, None]).to(out_ptr.dtype.element_ty),
            mask=valid_h[:, None],
        )
    elif ~valid_row:
        tl.store(
            out_ptr
            + (q_row * NUM_Q_HEADS + offs_h[:, None]) * HEAD_DIM
            + offs_d[None, :],
            0.0,
            mask=(q_row < TOTAL_Q) & valid_h[:, None],
        )


@libentry()
@triton.jit(do_not_specialize=["TOTAL_Q", "BATCH_SIZE"])
def _sparse_attn_sharedkv_decode_dynamic_cfa_dense_prefix_attention_kernel(
    q_ptr,
    compact_kv_ptr,
    compact_valid_ptr,
    cu_seqlens_q_ptr,
    seqused_kv_ptr,
    sinks_ptr,
    out_ptr,
    softmax_scale,
    TOTAL_Q,
    BATCH_SIZE,
    BATCH_BLOCK: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    MAX_COMPACT_TOKENS: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    CMP_BLOCK_N: tl.constexpr,
    HANDLE_ALIGNED: tl.constexpr,
):
    """CFA dense prefix：original/compressed 均按N64合并softmax状态。"""
    tl.static_assert(HEAD_DIM == 512, "DSV4 CFA dense prefix expects D=512.")
    tl.static_assert(CMP_BLOCK_N == 64, "DSV4 CFA dense prefix expects compressed N64.")

    pid = tl.program_id(0)
    head_blocks = tl.cdiv(NUM_Q_HEADS, HEAD_BLOCK)
    q_row = (pid // head_blocks).to(tl.int64)
    head_block_id = pid - q_row * head_blocks
    valid_row, ori_threshold, _safe_batch = _graphsafe_q_row_threshold(
        q_row,
        cu_seqlens_q_ptr,
        seqused_kv_ptr,
        BATCH_SIZE,
        BATCH_BLOCK,
    )
    cmp_count = tl.minimum(ori_threshold // 128, MAX_COMPACT_TOKENS - 128)
    compact_count = 128 + cmp_count  # noqa: F841
    all_valid = ori_threshold >= 128  # noqa: F841
    work_row = valid_row
    offs_h = head_block_id * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK).to(tl.int64)
    valid_h = work_row & (offs_h < NUM_Q_HEADS)
    offs_d = tl.arange(0, HEAD_DIM)

    if work_row:
        q = tl.load(
            q_ptr
            + (q_row * NUM_Q_HEADS + offs_h[:, None]) * HEAD_DIM
            + offs_d[None, :],
            mask=valid_h[:, None],
            other=0.0,
        )
        row_max = tl.load(sinks_ptr + offs_h, mask=valid_h, other=0.0).to(tl.float32)
        row_sum = tl.where(valid_h, 1.0, 0.0).to(tl.float32)
        acc = tl.zeros((HEAD_BLOCK, HEAD_DIM), dtype=tl.float32)

        offs_n64 = tl.arange(0, 64)
        for token_base in tl.static_range(0, 128, 64):
            ori_token = token_base + offs_n64
            valid_ori = tl.load(
                compact_valid_ptr + q_row * MAX_COMPACT_TOKENS + ori_token
            ).to(tl.int1)
            ori_tile = tl.load(
                compact_kv_ptr
                + ((q_row * MAX_COMPACT_TOKENS + ori_token)[:, None] * HEAD_DIM)
                + offs_d[None, :]
            )
            ori_scores = tl.dot(q, tl.trans(ori_tile)) * softmax_scale
            ori_scores = tl.where(
                valid_h[:, None] & valid_ori[None, :],
                ori_scores,
                float("-inf"),
            )
            ori_new_max = tl.maximum(row_max, tl.max(ori_scores, axis=1))
            ori_prob = tl.exp(ori_scores - ori_new_max[:, None])
            ori_prob = tl.where(valid_h[:, None] & valid_ori[None, :], ori_prob, 0.0)
            ori_alpha = tl.exp(row_max - ori_new_max)
            row_sum = row_sum * ori_alpha + tl.sum(ori_prob, axis=1)
            acc = acc * ori_alpha[:, None] + tl.dot(
                ori_prob.to(ori_tile.dtype), ori_tile
            )
            row_max = ori_new_max

        offs_n_cmp = tl.arange(0, CMP_BLOCK_N)
        for cmp_base in tl.range(0, cmp_count, CMP_BLOCK_N):
            cmp_pos = cmp_base + offs_n_cmp
            cmp_token = 128 + cmp_pos
            valid_cmp = tl.load(
                compact_valid_ptr + q_row * MAX_COMPACT_TOKENS + cmp_token,
                mask=cmp_pos < cmp_count,
                other=0,
            ).to(tl.int1)
            cmp_tile = tl.load(
                compact_kv_ptr
                + ((q_row * MAX_COMPACT_TOKENS + cmp_token)[:, None] * HEAD_DIM)
                + offs_d[None, :]
            )
            cmp_scores = tl.dot(q, tl.trans(cmp_tile)) * softmax_scale
            cmp_scores = tl.where(
                valid_h[:, None] & valid_cmp[None, :], cmp_scores, float("-inf")
            )
            cmp_new_max = tl.maximum(row_max, tl.max(cmp_scores, axis=1))
            cmp_prob = tl.exp(cmp_scores - cmp_new_max[:, None])
            cmp_prob = tl.where(valid_h[:, None] & valid_cmp[None, :], cmp_prob, 0.0)
            cmp_alpha = tl.exp(row_max - cmp_new_max)
            row_sum = row_sum * cmp_alpha + tl.sum(cmp_prob, axis=1)
            acc = acc * cmp_alpha[:, None] + tl.dot(
                cmp_prob.to(cmp_tile.dtype), cmp_tile
            )
            row_max = cmp_new_max

        tl.store(
            out_ptr
            + (q_row * NUM_Q_HEADS + offs_h[:, None]) * HEAD_DIM
            + offs_d[None, :],
            (acc / row_sum[:, None]).to(out_ptr.dtype.element_ty),
            mask=valid_h[:, None],
        )
    elif ~valid_row:
        tl.store(
            out_ptr
            + (q_row * NUM_Q_HEADS + offs_h[:, None]) * HEAD_DIM
            + offs_d[None, :],
            0.0,
            mask=(q_row < TOTAL_Q) & (offs_h[:, None] < NUM_Q_HEADS),
        )


@libentry()
@triton.jit(do_not_specialize=["TOTAL_Q", "BATCH_SIZE"])
def _sparse_attn_sharedkv_decode_dynamic_scfa_full_partial_kernel(
    q_ptr,
    compact_kv_ptr,
    compact_valid_ptr,
    cu_seqlens_q_ptr,
    seqused_kv_ptr,
    sinks_ptr,
    partial_acc_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    softmax_scale,
    TOTAL_Q,
    BATCH_SIZE,
    BATCH_BLOCK: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    MAX_COMPACT_TOKENS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    CHUNK_TOKENS: tl.constexpr,
    CHUNK_NUM: tl.constexpr,
):
    """SCFA 全有效 chunk：去掉逐 tile mask，使用 H32/N64。"""
    tl.static_assert(
        MAX_COMPACT_TOKENS == 640, "SCFA compact buffer expects 640 tokens."
    )
    tl.static_assert(
        CHUNK_NUM * CHUNK_TOKENS == MAX_COMPACT_TOKENS,
        "SCFA chunks must cover all 640 compact tokens.",
    )
    tl.static_assert(CHUNK_TOKENS % 64 == 0, "SCFA chunk size must be N64 aligned.")
    tl.static_assert(BLOCK_N == 64, "SCFA full partial expects N64.")

    pid = tl.program_id(0)
    head_blocks = tl.cdiv(NUM_Q_HEADS, HEAD_BLOCK)
    tasks_per_q = head_blocks * CHUNK_NUM
    q_row = (pid // tasks_per_q).to(tl.int64)
    local_task = pid - q_row * tasks_per_q
    head_block_id = (local_task // CHUNK_NUM).to(tl.int64)
    chunk_id = (local_task - head_block_id * CHUNK_NUM).to(tl.int64)
    valid_row, ori_threshold, _safe_batch = _graphsafe_q_row_threshold(
        q_row,
        cu_seqlens_q_ptr,
        seqused_kv_ptr,
        BATCH_SIZE,
        BATCH_BLOCK,
    )
    chunk_base = chunk_id * CHUNK_TOKENS
    compact_count = 128 + tl.minimum(ori_threshold // 4, 512)
    work_row = valid_row & (compact_count > 256)
    if work_row:
        offs_check = tl.arange(0, 64)
        valid_count = tl.zeros((), tl.int32)
        for check_base in tl.static_range(0, CHUNK_TOKENS, 64):
            check_token = chunk_base + check_base + offs_check
            chunk_valid = tl.load(
                compact_valid_ptr + q_row * MAX_COMPACT_TOKENS + check_token,
                mask=check_token < compact_count,
                other=0,
            ).to(tl.int32)
            valid_count += tl.sum(chunk_valid, axis=0)
        full_chunk = (chunk_base + CHUNK_TOKENS <= compact_count) & (
            valid_count == CHUNK_TOKENS
        )
        if full_chunk:
            offs_h = head_block_id * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK).to(tl.int64)
            valid_h = offs_h < NUM_Q_HEADS
            offs_d = tl.arange(0, HEAD_DIM)
            offs_n = tl.arange(0, BLOCK_N)
            q = tl.load(
                q_ptr
                + (q_row * NUM_Q_HEADS + offs_h[:, None]) * HEAD_DIM
                + offs_d[None, :],
                mask=valid_h[:, None],
                other=0.0,
            )
            if chunk_id == 0:
                row_max = tl.load(sinks_ptr + offs_h, mask=valid_h, other=0.0).to(
                    tl.float32
                )
                row_sum = tl.where(valid_h, 1.0, 0.0).to(tl.float32)
            else:
                row_max = tl.full((HEAD_BLOCK,), float("-inf"), tl.float32)
                row_sum = tl.zeros((HEAD_BLOCK,), tl.float32)
            acc = tl.zeros((HEAD_BLOCK, HEAD_DIM), dtype=tl.float32)

            for local_base in tl.static_range(0, CHUNK_TOKENS, BLOCK_N):
                token = chunk_base + local_base + offs_n
                kv = tl.load(
                    compact_kv_ptr
                    + ((q_row * MAX_COMPACT_TOKENS + token)[:, None] * HEAD_DIM)
                    + offs_d[None, :]
                )
                scores = tl.dot(q, tl.trans(kv)) * softmax_scale
                scores = tl.where(valid_h[:, None], scores, float("-inf"))
                new_max = tl.maximum(row_max, tl.max(scores, axis=1))
                p = tl.exp(scores - new_max[:, None])
                p = tl.where(valid_h[:, None], p, 0.0)
                alpha = tl.exp(row_max - new_max)
                row_sum = row_sum * alpha + tl.sum(p, axis=1)
                acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
                row_max = new_max

            partial_base = (q_row * CHUNK_NUM + chunk_id) * NUM_Q_HEADS + offs_h
            tl.store(partial_max_ptr + partial_base, row_max, mask=valid_h)
            tl.store(partial_sum_ptr + partial_base, row_sum, mask=valid_h)
            tl.store(
                partial_acc_ptr + partial_base[:, None] * HEAD_DIM + offs_d[None, :],
                acc,
                mask=valid_h[:, None],
            )


@libentry()
@triton.jit(do_not_specialize=["TOTAL_Q", "BATCH_SIZE"])
def _sparse_attn_sharedkv_decode_dynamic_scfa_partial_kernel(
    q_ptr,
    compact_kv_ptr,
    compact_valid_ptr,
    cu_seqlens_q_ptr,
    seqused_kv_ptr,
    sinks_ptr,
    partial_acc_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    softmax_scale,
    TOTAL_Q,
    BATCH_SIZE,
    BATCH_BLOCK: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    MAX_COMPACT_TOKENS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    CHUNK_TOKENS: tl.constexpr,
    CHUNK_NUM: tl.constexpr,
):
    """SCFA masked chunk fallback。

    program 轴只保留 (q_row, head_block)，五个 chunk 在 program 内顺序
    检查。full kernel 已处理的 chunk 不做计算；这样长序列常见的全有效
    情况不会为每个 chunk 再启动一组仅做判定的 program。
    """
    tl.static_assert(
        MAX_COMPACT_TOKENS == 640, "SCFA compact buffer expects 640 tokens."
    )
    tl.static_assert(
        CHUNK_NUM * CHUNK_TOKENS == MAX_COMPACT_TOKENS,
        "SCFA chunks must cover all 640 compact tokens.",
    )
    tl.static_assert(
        CHUNK_TOKENS % BLOCK_N == 0, "SCFA chunk size must be BLOCK_N aligned."
    )
    tl.static_assert(
        BLOCK_N == 32 or BLOCK_N == 64,
        "SCFA compact partial expects N32 or N64.",
    )

    pid = tl.program_id(0)
    head_blocks = tl.cdiv(NUM_Q_HEADS, HEAD_BLOCK)
    q_row = (pid // head_blocks).to(tl.int64)
    head_block_id = pid - q_row * head_blocks
    valid_row, ori_threshold, _safe_batch = _graphsafe_q_row_threshold(
        q_row,
        cu_seqlens_q_ptr,
        seqused_kv_ptr,
        BATCH_SIZE,
        BATCH_BLOCK,
    )
    offs_h = head_block_id * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK).to(tl.int64)
    valid_h = valid_row & (offs_h < NUM_Q_HEADS)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_n = tl.arange(0, BLOCK_N)

    compact_count = 128 + tl.minimum(ori_threshold // 4, 512)
    work_row = valid_row & (compact_count > 256)

    if work_row:
        q = tl.load(
            q_ptr
            + (q_row * NUM_Q_HEADS + offs_h[:, None]) * HEAD_DIM
            + offs_d[None, :],
            mask=valid_h[:, None],
            other=0.0,
        )
        offs_chunk = tl.arange(0, CHUNK_TOKENS)
        for chunk_id in tl.range(0, CHUNK_NUM):
            chunk_base = chunk_id * CHUNK_TOKENS
            chunk_has_tokens = chunk_base < compact_count
            valid_chunk_slot = chunk_base + offs_chunk < compact_count
            chunk_valid = tl.load(
                compact_valid_ptr
                + q_row * MAX_COMPACT_TOKENS
                + chunk_base
                + offs_chunk,
                mask=chunk_has_tokens & valid_chunk_slot,
                other=0,
            ).to(tl.int32)
            full_chunk = (
                chunk_has_tokens
                & (chunk_base + CHUNK_TOKENS <= compact_count)
                & (tl.sum(chunk_valid, axis=0) == CHUNK_TOKENS)
            )

            if chunk_has_tokens & (~full_chunk):
                if chunk_id == 0:
                    row_max = tl.load(sinks_ptr + offs_h, mask=valid_h, other=0.0).to(
                        tl.float32
                    )
                    row_sum = tl.where(valid_h, 1.0, 0.0).to(tl.float32)
                else:
                    row_max = tl.full((HEAD_BLOCK,), float("-inf"), tl.float32)
                    row_sum = tl.zeros((HEAD_BLOCK,), tl.float32)
                acc = tl.zeros((HEAD_BLOCK, HEAD_DIM), dtype=tl.float32)

                for local_base in tl.static_range(0, CHUNK_TOKENS, BLOCK_N):
                    token = chunk_base + local_base + offs_n
                    token_in_range = token < compact_count
                    valid_token = tl.load(
                        compact_valid_ptr + q_row * MAX_COMPACT_TOKENS + token,
                        mask=token_in_range,
                        other=0,
                    ).to(tl.int1)
                    kv = tl.load(
                        compact_kv_ptr
                        + ((q_row * MAX_COMPACT_TOKENS + token)[:, None] * HEAD_DIM)
                        + offs_d[None, :],
                        mask=valid_token[:, None],
                        other=0.0,
                    )
                    scores = tl.dot(q, tl.trans(kv)) * softmax_scale
                    scores = tl.where(
                        valid_h[:, None] & valid_token[None, :], scores, float("-inf")
                    )
                    has_token = tl.sum(valid_token.to(tl.int32), axis=0) > 0
                    tile_max = tl.max(scores, axis=1)
                    new_max = tl.where(
                        has_token, tl.maximum(row_max, tile_max), row_max
                    )
                    p = tl.exp(scores - new_max[:, None])
                    p = tl.where(valid_h[:, None] & valid_token[None, :], p, 0.0)
                    alpha = tl.where(has_token, tl.exp(row_max - new_max), 1.0)
                    row_sum = row_sum * alpha + tl.sum(p, axis=1)
                    acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
                    row_max = new_max

                partial_base = (q_row * CHUNK_NUM + chunk_id) * NUM_Q_HEADS + offs_h
                tl.store(partial_max_ptr + partial_base, row_max, mask=valid_h)
                tl.store(partial_sum_ptr + partial_base, row_sum, mask=valid_h)
                tl.store(
                    partial_acc_ptr
                    + partial_base[:, None] * HEAD_DIM
                    + offs_d[None, :],
                    acc,
                    mask=valid_h[:, None],
                )
            elif ~chunk_has_tokens:
                partial_base = (q_row * CHUNK_NUM + chunk_id) * NUM_Q_HEADS + offs_h
                tl.store(partial_max_ptr + partial_base, float("-inf"), mask=valid_h)
                tl.store(partial_sum_ptr + partial_base, 0.0, mask=valid_h)
                tl.store(
                    partial_acc_ptr
                    + partial_base[:, None] * HEAD_DIM
                    + offs_d[None, :],
                    0.0,
                    mask=valid_h[:, None],
                )


@libentry()
@triton.jit(do_not_specialize=["TOTAL_Q", "BATCH_SIZE"])
def _sparse_attn_sharedkv_decode_dynamic_scfa_merge_kernel(
    q_ptr,
    compact_kv_ptr,
    compact_valid_ptr,
    sinks_ptr,
    partial_acc_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    cu_seqlens_q_ptr,
    seqused_kv_ptr,
    out_ptr,
    softmax_scale,
    TOTAL_Q,
    BATCH_SIZE,
    BATCH_BLOCK: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    MAX_COMPACT_TOKENS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CHUNK_NUM: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    SHORT_COMPACT_LIMIT: tl.constexpr,
    DIRECT_ALL: tl.constexpr,
):
    """长行合并五个 partial；短行直接计算并写最终 O。"""
    pid = tl.program_id(0)
    head_blocks = tl.cdiv(NUM_Q_HEADS, HEAD_BLOCK)
    q_row = (pid // head_blocks).to(tl.int64)
    head_block_id = pid - q_row * head_blocks
    valid_row, ori_threshold, _safe_batch = _graphsafe_q_row_threshold(
        q_row,
        cu_seqlens_q_ptr,
        seqused_kv_ptr,
        BATCH_SIZE,
        BATCH_BLOCK,
    )
    compact_count = 128 + tl.minimum(ori_threshold // 4, 512)
    work_row = valid_row & (compact_count > 256)
    offs_h = head_block_id.to(tl.int64) * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK).to(
        tl.int64
    )
    valid_h = offs_h < NUM_Q_HEADS
    offs_d = tl.arange(0, HEAD_DIM)

    if work_row and not DIRECT_ALL:
        row_max = tl.full((HEAD_BLOCK,), float("-inf"), tl.float32)
        for chunk in tl.static_range(0, CHUNK_NUM):
            base = (q_row * CHUNK_NUM + chunk) * NUM_Q_HEADS
            row_max = tl.maximum(
                row_max,
                tl.load(
                    partial_max_ptr + base + offs_h, mask=valid_h, other=float("-inf")
                ),
            )

        row_sum = tl.zeros((HEAD_BLOCK,), tl.float32)
        acc = tl.zeros((HEAD_BLOCK, HEAD_DIM), tl.float32)
        for chunk in tl.static_range(0, CHUNK_NUM):
            base = (q_row * CHUNK_NUM + chunk) * NUM_Q_HEADS
            chunk_max = tl.load(
                partial_max_ptr + base + offs_h, mask=valid_h, other=float("-inf")
            )
            scale = tl.exp(chunk_max - row_max)
            row_sum += (
                tl.load(partial_sum_ptr + base + offs_h, mask=valid_h, other=0.0)
                * scale
            )
            acc += (
                tl.load(
                    partial_acc_ptr
                    + (base + offs_h[:, None]) * HEAD_DIM
                    + offs_d[None, :],
                    mask=valid_h[:, None],
                    other=0.0,
                )
                * scale[:, None]
            )

        tl.store(
            out_ptr
            + (q_row * NUM_Q_HEADS + offs_h[:, None]) * HEAD_DIM
            + offs_d[None, :],
            (acc / row_sum[:, None]).to(out_ptr.dtype.element_ty),
            mask=valid_h[:, None],
        )
    elif valid_row:
        q = tl.load(
            q_ptr
            + (q_row * NUM_Q_HEADS + offs_h[:, None]) * HEAD_DIM
            + offs_d[None, :],
            mask=valid_h[:, None],
            other=0.0,
        )
        row_max = tl.load(sinks_ptr + offs_h, mask=valid_h, other=0.0).to(tl.float32)
        row_sum = tl.where(valid_h, 1.0, 0.0).to(tl.float32)
        acc = tl.zeros((HEAD_BLOCK, HEAD_DIM), tl.float32)
        offs_n = tl.arange(0, BLOCK_N)
        for token_base in tl.range(0, compact_count, BLOCK_N):
            token = token_base + offs_n
            valid_token = tl.load(
                compact_valid_ptr + q_row * MAX_COMPACT_TOKENS + token,
                mask=token < compact_count,
                other=0,
            ).to(tl.int1)
            kv = tl.load(
                compact_kv_ptr
                + ((q_row * MAX_COMPACT_TOKENS + token)[:, None] * HEAD_DIM)
                + offs_d[None, :]
            )
            scores = tl.dot(q, tl.trans(kv)) * softmax_scale
            scores = tl.where(
                valid_h[:, None] & valid_token[None, :], scores, float("-inf")
            )
            new_max = tl.maximum(row_max, tl.max(scores, axis=1))
            p = tl.exp(scores - new_max[:, None])
            p = tl.where(valid_h[:, None] & valid_token[None, :], p, 0.0)
            alpha = tl.exp(row_max - new_max)
            row_sum = row_sum * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
            row_max = new_max
        tl.store(
            out_ptr
            + (q_row * NUM_Q_HEADS + offs_h[:, None]) * HEAD_DIM
            + offs_d[None, :],
            (acc / row_sum[:, None]).to(out_ptr.dtype.element_ty),
            mask=valid_h[:, None],
        )
    else:
        tl.store(
            out_ptr
            + (q_row * NUM_Q_HEADS + offs_h[:, None]) * HEAD_DIM
            + offs_d[None, :],
            0.0,
            mask=(q_row < TOTAL_Q) & valid_h[:, None],
        )


@libentry()
@triton.jit
def _sparse_attn_sharedkv_short_single_prefill_fused_kernel(
    q_ptr,
    ori_kv_ptr,
    cmp_kv_ptr,
    cmp_sparse_indices_ptr,
    ori_block_table_ptr,
    cmp_block_table_ptr,
    cu_seqlens_q_ptr,
    seqused_kv_ptr,
    sinks_ptr,
    out_ptr,
    softmax_scale,
    TOTAL_Q_CAPACITY: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CMP_BLOCK_N: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    MODE: tl.constexpr,
    CMP_RATIO: tl.constexpr,
):
    """Fuse PA reads and attention for a single short prefill sequence."""
    tl.static_assert(
        TOTAL_Q_CAPACITY <= 16, "Short prefill supports at most 16 Q rows."
    )
    tl.static_assert(HEAD_DIM == 512, "Short prefill expects D=512.")
    tl.static_assert(PAGE_SIZE == 128, "Short prefill expects N128 PA pages.")
    tl.static_assert(
        BLOCK_N == 32 or BLOCK_N == 64, "Short prefill uses N32/N64 tiles."
    )
    tl.static_assert(
        CMP_BLOCK_N == BLOCK_N or CMP_BLOCK_N == 8,
        "Short SCFA compressed tiles use N8 or the main tile width.",
    )
    tl.static_assert(MODE == 0 or MODE == 1 or MODE == 2, "Invalid short-prefill mode.")

    pid = tl.program_id(0)
    head_blocks = tl.cdiv(NUM_Q_HEADS, HEAD_BLOCK)
    q_row = (pid // head_blocks).to(tl.int64)
    head_block_id = pid - q_row * head_blocks
    active_q = tl.load(cu_seqlens_q_ptr + 1).to(tl.int64)
    kv_len = tl.load(seqused_kv_ptr).to(tl.int64)
    valid_row = (q_row < active_q) & (kv_len <= PAGE_SIZE)
    ori_threshold = kv_len - active_q + q_row + 1

    offs_h = head_block_id.to(tl.int64) * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK).to(
        tl.int64
    )
    valid_h = offs_h < NUM_Q_HEADS
    offs_d = tl.arange(0, HEAD_DIM)
    offs_n = tl.arange(0, BLOCK_N).to(tl.int64)
    cmp_offs_n = tl.arange(0, CMP_BLOCK_N).to(tl.int64)
    if valid_row:
        q = tl.load(
            q_ptr
            + (q_row * NUM_Q_HEADS + offs_h[:, None]) * HEAD_DIM
            + offs_d[None, :],
            mask=valid_h[:, None],
            other=0.0,
        )
        row_max = tl.load(sinks_ptr + offs_h, mask=valid_h, other=0.0).to(tl.float32)
        row_sum = tl.where(valid_h, 1.0, 0.0).to(tl.float32)
        acc = tl.zeros((HEAD_BLOCK, HEAD_DIM), tl.float32)

        ori_physical_page = tl.load(ori_block_table_ptr).to(tl.int64)
        for token_base in tl.static_range(0, PAGE_SIZE, BLOCK_N):
            token = token_base + offs_n
            valid_token = token < ori_threshold
            kv = tl.load(
                ori_kv_ptr
                + (ori_physical_page * PAGE_SIZE + token[:, None]) * HEAD_DIM
                + offs_d[None, :]
            )
            scores = tl.dot(q, tl.trans(kv)) * softmax_scale
            scores = tl.where(
                valid_h[:, None] & valid_token[None, :], scores, float("-inf")
            )
            new_max = tl.maximum(row_max, tl.max(scores, axis=1))
            p = tl.exp(scores - new_max[:, None])
            p = tl.where(valid_h[:, None] & valid_token[None, :], p, 0.0)
            alpha = tl.exp(row_max - new_max)
            row_sum = row_sum * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
            row_max = new_max

        if MODE != 0:
            cmp_threshold = ori_threshold // CMP_RATIO
            cmp_count = tl.minimum(cmp_threshold, CMP_BLOCK_N)
            cmp_slot = cmp_offs_n
            valid_cmp = cmp_slot < cmp_count
            if MODE == 1:
                logical_cmp_pos = cmp_slot
            else:
                sparse_idx = tl.load(
                    cmp_sparse_indices_ptr + q_row * 512 + cmp_slot,
                    mask=valid_cmp,
                    other=-1,
                ).to(tl.int64)
                valid_cmp &= (sparse_idx >= 0) & (sparse_idx < cmp_threshold)
                logical_cmp_pos = tl.maximum(sparse_idx, 0)
            cmp_page = logical_cmp_pos // PAGE_SIZE
            cmp_offset = logical_cmp_pos - cmp_page * PAGE_SIZE
            cmp_physical_page = tl.load(
                cmp_block_table_ptr + cmp_page,
                mask=valid_cmp,
                other=0,
            ).to(tl.int64)
            cmp_kv = tl.load(
                cmp_kv_ptr
                + (cmp_physical_page[:, None] * PAGE_SIZE + cmp_offset[:, None])
                * HEAD_DIM
                + offs_d[None, :]
            )
            scores = tl.dot(q, tl.trans(cmp_kv)) * softmax_scale
            scores = tl.where(
                valid_h[:, None] & valid_cmp[None, :], scores, float("-inf")
            )
            new_max = tl.maximum(row_max, tl.max(scores, axis=1))
            p = tl.exp(scores - new_max[:, None])
            p = tl.where(valid_h[:, None] & valid_cmp[None, :], p, 0.0)
            alpha = tl.exp(row_max - new_max)
            row_sum = row_sum * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(cmp_kv.dtype), cmp_kv)

        tl.store(
            out_ptr
            + (q_row * NUM_Q_HEADS + offs_h[:, None]) * HEAD_DIM
            + offs_d[None, :],
            (acc / row_sum[:, None]).to(out_ptr.dtype.element_ty),
            mask=valid_h[:, None],
        )
    else:
        tl.store(
            out_ptr
            + (q_row * NUM_Q_HEADS + offs_h[:, None]) * HEAD_DIM
            + offs_d[None, :],
            0.0,
            mask=(q_row < TOTAL_Q_CAPACITY) & valid_h[:, None],
        )


@libentry()
@triton.jit(do_not_specialize=["BATCH_SIZE"])
def _sparse_attn_sharedkv_graphsafe_q1_fused_kernel(
    q_ptr,
    ori_kv_ptr,
    cmp_kv_ptr,
    cmp_sparse_indices_ptr,
    ori_block_table_ptr,
    cmp_block_table_ptr,
    cu_seqlens_q_ptr,
    seqused_kv_ptr,
    sinks_ptr,
    out_ptr,
    softmax_scale,
    BATCH_SIZE,
    BATCH_BLOCK: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    ORI_TABLE_WIDTH: tl.constexpr,
    CMP_TABLE_WIDTH: tl.constexpr,
    MAX_CMP_TOKENS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    MODE: tl.constexpr,
    CMP_RATIO: tl.constexpr,
):
    """Direct PA attention for the q1 Graph bucket with dynamic KV length."""
    tl.static_assert(HEAD_DIM == 512, "Graph q1 expects D=512.")
    tl.static_assert(PAGE_SIZE == 128, "Graph q1 expects N128 PA pages.")
    tl.static_assert(BLOCK_N == 64, "Graph q1 uses N64 attention tiles.")
    tl.static_assert(MODE == 0 or MODE == 1 or MODE == 2, "Invalid Graph q1 mode.")

    head_block_id = tl.program_id(0).to(tl.int64)
    valid_row, ori_threshold, safe_batch = _graphsafe_q_row_threshold(
        0,
        cu_seqlens_q_ptr,
        seqused_kv_ptr,
        BATCH_SIZE,
        BATCH_BLOCK,
    )
    offs_h = head_block_id * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK).to(tl.int64)
    valid_h = offs_h < NUM_Q_HEADS
    offs_d = tl.arange(0, HEAD_DIM)
    offs_n = tl.arange(0, BLOCK_N).to(tl.int64)
    if valid_row:
        q = tl.load(
            q_ptr + offs_h[:, None] * HEAD_DIM + offs_d[None, :],
            mask=valid_h[:, None],
            other=0.0,
        )
        row_max = tl.load(sinks_ptr + offs_h, mask=valid_h, other=0.0).to(tl.float32)
        row_sum = tl.where(valid_h, 1.0, 0.0).to(tl.float32)
        acc = tl.zeros((HEAD_BLOCK, HEAD_DIM), tl.float32)

        for token_base in tl.static_range(0, 128, BLOCK_N):
            ori_pos = ori_threshold - 128 + token_base + offs_n
            valid_token = (ori_pos >= 0) & (ori_pos < ori_threshold)
            safe_ori_pos = tl.maximum(ori_pos, 0)
            ori_page = safe_ori_pos // PAGE_SIZE
            ori_offset = safe_ori_pos - ori_page * PAGE_SIZE
            valid_token &= ori_page < ORI_TABLE_WIDTH
            ori_physical_page = tl.load(
                ori_block_table_ptr + safe_batch * ORI_TABLE_WIDTH + ori_page,
                mask=valid_token,
                other=0,
            ).to(tl.int64)
            valid_token &= ori_physical_page >= 0
            kv = tl.load(
                ori_kv_ptr
                + (
                    tl.maximum(ori_physical_page, 0)[:, None] * PAGE_SIZE
                    + ori_offset[:, None]
                )
                * HEAD_DIM
                + offs_d[None, :]
            )
            scores = tl.dot(q, tl.trans(kv)) * softmax_scale
            scores = tl.where(
                valid_h[:, None] & valid_token[None, :], scores, float("-inf")
            )
            new_max = tl.maximum(row_max, tl.max(scores, axis=1))
            p = tl.exp(scores - new_max[:, None])
            p = tl.where(valid_h[:, None] & valid_token[None, :], p, 0.0)
            alpha = tl.exp(row_max - new_max)
            row_sum = row_sum * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
            row_max = new_max

        if MODE != 0:
            cmp_threshold = ori_threshold // CMP_RATIO
            cmp_count = tl.minimum(cmp_threshold, MAX_CMP_TOKENS)
            for cmp_base in tl.range(0, cmp_count, BLOCK_N):
                cmp_slot = cmp_base + offs_n
                valid_cmp = cmp_slot < cmp_count
                if MODE == 1:
                    logical_cmp_pos = cmp_slot
                else:
                    sparse_idx = tl.load(
                        cmp_sparse_indices_ptr + cmp_slot,
                        mask=valid_cmp,
                        other=-1,
                    ).to(tl.int64)
                    valid_cmp &= (sparse_idx >= 0) & (sparse_idx < cmp_threshold)
                    logical_cmp_pos = tl.maximum(sparse_idx, 0)
                cmp_page = logical_cmp_pos // PAGE_SIZE
                cmp_offset = logical_cmp_pos - cmp_page * PAGE_SIZE
                valid_cmp &= cmp_page < CMP_TABLE_WIDTH
                cmp_physical_page = tl.load(
                    cmp_block_table_ptr + safe_batch * CMP_TABLE_WIDTH + cmp_page,
                    mask=valid_cmp,
                    other=0,
                ).to(tl.int64)
                valid_cmp &= cmp_physical_page >= 0
                cmp_kv = tl.load(
                    cmp_kv_ptr
                    + (
                        tl.maximum(cmp_physical_page, 0)[:, None] * PAGE_SIZE
                        + cmp_offset[:, None]
                    )
                    * HEAD_DIM
                    + offs_d[None, :]
                )
                scores = tl.dot(q, tl.trans(cmp_kv)) * softmax_scale
                scores = tl.where(
                    valid_h[:, None] & valid_cmp[None, :], scores, float("-inf")
                )
                new_max = tl.maximum(row_max, tl.max(scores, axis=1))
                p = tl.exp(scores - new_max[:, None])
                p = tl.where(valid_h[:, None] & valid_cmp[None, :], p, 0.0)
                alpha = tl.exp(row_max - new_max)
                row_sum = row_sum * alpha + tl.sum(p, axis=1)
                acc = acc * alpha[:, None] + tl.dot(p.to(cmp_kv.dtype), cmp_kv)
                row_max = new_max

        tl.store(
            out_ptr + offs_h[:, None] * HEAD_DIM + offs_d[None, :],
            (acc / row_sum[:, None]).to(out_ptr.dtype.element_ty),
            mask=valid_h[:, None],
        )
    else:
        tl.store(
            out_ptr + offs_h[:, None] * HEAD_DIM + offs_d[None, :],
            0.0,
            mask=valid_h[:, None],
        )


def _launch_graphsafe_dynamic_compact_decode(
    q: torch.Tensor,
    *,
    ori_kv: torch.Tensor,
    cmp_kv: torch.Tensor,
    cmp_sparse_indices: torch.Tensor,
    ori_block_table: torch.Tensor,
    cmp_block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_kv: torch.Tensor,
    sinks: torch.Tensor,
    softmax_scale: float,
    cmp_ratio: int,
    mode: int,
    batch_size: int,
    max_model_len: Optional[int],
    eager_known_max_kv: Optional[int],
) -> torch.Tensor:
    """启动可被 ACL Graph 捕获的动态 compact-pack decode 路径。"""
    global _LAST_GRAPHSAFE_WORKSPACE

    total_q = int(q.shape[0])
    # The adapter supplies known lengths only outside Graph capture.
    if total_q == 1 and eager_known_max_kv is not None:
        out = torch.empty_like(q)
        head_block = 8
        batch_block = max(2, triton.next_power_of_2(batch_size))
        if mode == 0:
            max_cmp_tokens = 0
        elif mode == 1:
            max_cmp_tokens = min(
                int(cmp_block_table.shape[1]) * 128,
                math.ceil(int(eager_known_max_kv) / int(cmp_ratio)),
            )
            max_cmp_tokens = max(64, triton.cdiv(max_cmp_tokens, 64) * 64)
        else:
            max_cmp_tokens = 512
        kernel_entry = _sparse_attn_sharedkv_graphsafe_q1_fused_kernel
        runtime_args = (
            q,
            ori_kv,
            cmp_kv,
            cmp_sparse_indices,
            ori_block_table,
            cmp_block_table,
            cu_seqlens_q,
            seqused_kv,
            sinks,
            out,
            float(softmax_scale),
            batch_size,
        )
        constexpr_names = (
            "BATCH_BLOCK",
            "NUM_Q_HEADS",
            "HEAD_DIM",
            "PAGE_SIZE",
            "ORI_TABLE_WIDTH",
            "CMP_TABLE_WIDTH",
            "MAX_CMP_TOKENS",
            "BLOCK_N",
            "HEAD_BLOCK",
            "MODE",
            "CMP_RATIO",
        )
        constexpr_values = (
            batch_block,
            64,
            512,
            128,
            int(ori_block_table.shape[1]),
            int(cmp_block_table.shape[1]),
            max_cmp_tokens,
            64,
            head_block,
            mode,
            int(cmp_ratio),
        )
        dispatch_key = (
            *(tensor.dtype for tensor in runtime_args[:10]),
            *(
                tensor.data_ptr() % kernel_entry.divisibility == 0
                for tensor in runtime_args[:10]
            ),
            float(softmax_scale),
            *constexpr_values,
        )
        _launch_cached_eager_kernel(
            kernel_entry,
            (triton.cdiv(64, head_block),),
            runtime_args,
            constexpr_names,
            constexpr_values,
            _prefill_launch_options(multibuffer=False, unit_flag=True),
            dispatch_key,
        )
        _LAST_GRAPHSAFE_WORKSPACE = ()
        return out

    if (
        batch_size == 1
        and total_q <= 16
        and eager_known_max_kv is not None
        and eager_known_max_kv <= 128
    ):
        out = torch.empty_like(q)
        head_block = _MODEL_SHORT_SINGLE_PREFILL_HEAD_BLOCK
        # N64 reduces the original/SWA and dense-CFA tile count, while SCFA
        # keeps N32 to avoid carrying masked compressed lanes through tl.dot.
        block_n = (
            _MODEL_SHORT_SINGLE_PREFILL_BLOCK_N
            if mode == 2
            else _MODEL_SHORT_SINGLE_PREFILL_WIDE_BLOCK_N
        )
        # q16/KV32 SCFA has at most eight compressed tokens. Keep the
        # original attention tile at N32 but avoid masked compressed dot lanes.
        cmp_block_n = (
            _MODEL_SHORT_SINGLE_PREFILL_SCFA_SHORT_CMP_BLOCK_N
            if mode == 2 and eager_known_max_kv <= 32
            else block_n
        )
        grid = total_q * triton.cdiv(64, head_block)
        kernel_entry = _sparse_attn_sharedkv_short_single_prefill_fused_kernel
        runtime_args = (
            q,
            ori_kv,
            cmp_kv,
            cmp_sparse_indices,
            ori_block_table,
            cmp_block_table,
            cu_seqlens_q,
            seqused_kv,
            sinks,
            out,
            float(softmax_scale),
        )
        constexpr_names = (
            "TOTAL_Q_CAPACITY",
            "NUM_Q_HEADS",
            "HEAD_DIM",
            "PAGE_SIZE",
            "BLOCK_N",
            "CMP_BLOCK_N",
            "HEAD_BLOCK",
            "MODE",
            "CMP_RATIO",
        )
        constexpr_values = (
            total_q,
            64,
            512,
            128,
            block_n,
            cmp_block_n,
            head_block,
            mode,
            int(cmp_ratio),
        )
        dispatch_key = (
            *(tensor.dtype for tensor in runtime_args[:-1]),
            *(
                tensor.data_ptr() % kernel_entry.divisibility == 0
                for tensor in runtime_args[:-1]
            ),
            float(softmax_scale),
            *constexpr_values,
        )
        _launch_cached_eager_kernel(
            kernel_entry,
            (grid,),
            runtime_args,
            constexpr_names,
            constexpr_values,
            _prefill_launch_options(multibuffer=False, unit_flag=True),
            dispatch_key,
        )
        _LAST_GRAPHSAFE_WORKSPACE = ()
        return out

    # FlagTree 对长度为 1 的向量归约会在 TritonToLinalg 阶段留下
    # tensor<1> -> scalar 的未决 materialization。并发 3..8 统一使用宽度 8，
    # 避免请求逐步完成时为宽度 4 重新 JIT；多余 lane 由 BATCH_SIZE mask 掉。
    if batch_size <= 2:
        batch_block = 2
    elif batch_size <= 8:
        batch_block = 8
    else:
        batch_block = triton.next_power_of_2(batch_size)
    if mode == 0:
        max_cmp_tokens = 0
        max_compact_tokens = 128
    elif mode == 1:
        table_capacity = int(cmp_block_table.shape[1]) * 128
        if max_model_len is None:
            max_cmp_tokens = table_capacity
        else:
            max_cmp_tokens = min(
                table_capacity, math.ceil(int(max_model_len) / int(cmp_ratio))
            )
        # N64 对齐让主计算最后一个 tile 只需要 valid mask，不改变数学语义。
        max_cmp_tokens = max(64, triton.cdiv(max_cmp_tokens, 64) * 64)
        max_compact_tokens = 128 + max_cmp_tokens
    else:
        max_cmp_tokens = 512
        max_compact_tokens = 640

    out = torch.empty_like(q)
    dynamic_options = _prefill_launch_options(multibuffer=False, unit_flag=True)
    compact_shape = (total_q, max_compact_tokens, 512)
    valid_shape = (total_q, max_compact_tokens)
    if eager_known_max_kv is None:
        compact_kv = torch.empty(compact_shape, dtype=ori_kv.dtype, device=q.device)
        compact_valid = torch.empty(valid_shape, dtype=torch.uint8, device=q.device)
    else:
        # FULL_DECODE_ONLY的prefill不在Graph内，同模式各层按同一stream顺序
        # 执行；复用稳定workspace可避免每层重新分配大块compact GM。
        compact_kv = _eager_workspace(
            f"graphsafe.mode{mode}.compact_kv",
            compact_shape,
            dtype=ori_kv.dtype,
            device=q.device,
        )
        compact_valid = _eager_workspace(
            f"graphsafe.mode{mode}.compact_valid",
            valid_shape,
            dtype=torch.uint8,
            device=q.device,
        )
    pack_options = dict(dynamic_options)
    pack_options["disable_auto_inject_block_sync"] = True
    # Graph pack 不再在一个 program 内使用动态循环：original 固定两个
    # N64 program；compressed 每个 N64 tile 一个 program。
    # program 只执行 mask，不搬运数据，换取重复 replay 时固定的控制流。
    original_tiles_per_q = 2
    fused_q1_scfa = total_q == 1 and mode == 2
    fused_q8_scfa = total_q == 2 and mode == 2
    # The same fused pack kernel also covers the q8 CFA eager bucket. Its row
    # metadata is still device-derived, so known eager lengths do not change
    # the pack semantics; q1/q2 keep their existing capture-only guard.
    fused_q1_q2_cfa = mode == 1 and (
        (total_q in (1, 2) and eager_known_max_kv is None) or total_q == 8
    )
    if fused_q1_q2_cfa:
        _sparse_attn_sharedkv_q1_q2_cfa_fused_pack_kernel[(total_q * 3,)](
            ori_kv,
            cmp_kv,
            ori_block_table,
            cmp_block_table,
            cu_seqlens_q,
            seqused_kv,
            compact_kv,
            compact_valid,
            TOTAL_Q=total_q,
            BATCH_SIZE=batch_size,
            BATCH_BLOCK=batch_block,
            ORI_TABLE_WIDTH=int(ori_block_table.shape[1]),
            CMP_TABLE_WIDTH=int(cmp_block_table.shape[1]),
            MAX_CMP_TOKENS=max_cmp_tokens,
            MAX_COMPACT_TOKENS=max_compact_tokens,
            **pack_options,
        )
    elif fused_q1_scfa:
        _sparse_attn_sharedkv_q1_scfa_pack_kernel[(10,)](
            ori_kv,
            cmp_kv,
            cmp_sparse_indices,
            ori_block_table,
            cmp_block_table,
            cu_seqlens_q,
            seqused_kv,
            compact_kv,
            compact_valid,
            BATCH_SIZE=batch_size,
            BATCH_BLOCK=batch_block,
            ORI_TABLE_WIDTH=int(ori_block_table.shape[1]),
            CMP_TABLE_WIDTH=int(cmp_block_table.shape[1]),
            **pack_options,
        )
    elif fused_q8_scfa:
        _sparse_attn_sharedkv_q8_scfa_pack_kernel[(total_q * 10,)](
            ori_kv,
            cmp_kv,
            cmp_sparse_indices,
            ori_block_table,
            cmp_block_table,
            cu_seqlens_q,
            seqused_kv,
            compact_kv,
            compact_valid,
            BATCH_SIZE=batch_size,
            BATCH_BLOCK=batch_block,
            ORI_TABLE_WIDTH=int(ori_block_table.shape[1]),
            CMP_TABLE_WIDTH=int(cmp_block_table.shape[1]),
            **pack_options,
        )
    else:
        _sparse_attn_sharedkv_decode_dynamic_pack_original_n64_kernel[
            (total_q * original_tiles_per_q,)
        ](
            ori_kv,
            ori_block_table,
            cu_seqlens_q,
            seqused_kv,
            compact_kv,
            compact_valid,
            TOTAL_Q=total_q,
            BATCH_SIZE=batch_size,
            BATCH_BLOCK=batch_block,
            HEAD_DIM=512,
            ORI_BLOCK_SIZE=128,
            ORI_TABLE_WIDTH=int(ori_block_table.shape[1]),
            MAX_COMPACT_TOKENS=max_compact_tokens,
            TILES_PER_Q=original_tiles_per_q,
            BLOCK_N=64,
            BLOCK_D=512,
            **pack_options,
        )
    if mode == 1 and not fused_q1_q2_cfa:
        _sparse_attn_sharedkv_decode_dynamic_pack_cfa_rows_kernel[(total_q,)](
            cmp_kv,
            cmp_block_table,
            cu_seqlens_q,
            seqused_kv,
            compact_kv,
            compact_valid,
            TOTAL_Q=total_q,
            BATCH_SIZE=batch_size,
            BATCH_BLOCK=batch_block,
            HEAD_DIM=512,
            CMP_BLOCK_SIZE=128,
            CMP_TABLE_WIDTH=int(cmp_block_table.shape[1]),
            CMP_RATIO=int(cmp_ratio),
            MAX_CMP_TOKENS=max_cmp_tokens,
            MAX_COMPACT_TOKENS=max_compact_tokens,
            BLOCK_N=64,
            BLOCK_D=512,
            **pack_options,
        )
    elif mode == 2 and not (fused_q1_scfa or fused_q8_scfa):
        cmp_tiles_per_q = triton.cdiv(max_cmp_tokens, 64)
        _sparse_attn_sharedkv_decode_dynamic_pack_compressed_n64_kernel[
            (total_q * cmp_tiles_per_q,)
        ](
            cmp_kv,
            cmp_sparse_indices,
            cmp_block_table,
            cu_seqlens_q,
            seqused_kv,
            compact_kv,
            compact_valid,
            TOTAL_Q=total_q,
            BATCH_SIZE=batch_size,
            BATCH_BLOCK=batch_block,
            HEAD_DIM=512,
            CMP_BLOCK_SIZE=128,
            CMP_TABLE_WIDTH=int(cmp_block_table.shape[1]),
            CMP_RATIO=int(cmp_ratio),
            MODE=mode,
            MAX_CMP_TOKENS=max_cmp_tokens,
            MAX_COMPACT_TOKENS=max_compact_tokens,
            TILES_PER_Q=cmp_tiles_per_q,
            BLOCK_N=64,
            BLOCK_D=512,
            **pack_options,
        )

    if mode != 2:
        if mode == 0:
            fast_head_block = 16
            fast_grid = total_q * triton.cdiv(64, fast_head_block)
            _sparse_attn_sharedkv_decode_dynamic_compact_full_attention_kernel[
                (fast_grid,)
            ](
                q,
                compact_kv,
                compact_valid,
                cu_seqlens_q,
                seqused_kv,
                sinks,
                out,
                float(softmax_scale),
                TOTAL_Q=total_q,
                BATCH_SIZE=batch_size,
                BATCH_BLOCK=batch_block,
                NUM_Q_HEADS=64,
                HEAD_DIM=512,
                MAX_COMPACT_TOKENS=max_compact_tokens,
                BLOCK_N=64,
                HEAD_BLOCK=fast_head_block,
                MODE=mode,
                STATIC_SWA_LOOP=mode == 0,
            )
        else:
            # q1用H8提供足够并行度。独立q8/batch8测试中H32可编译，但模型
            # FULL_DECODE_ONLY会特化到batch32，额外的batch查找状态使UB需求
            # 达到1,612,288 bit并超过910B的1,572,864 bit，因此模型bucket
            # 统一使用H16。它会增加program数，但保证所有Graph容量可编译。
            dense_head_block = 8 if total_q == 1 else 16
            dense_grid = total_q * triton.cdiv(64, dense_head_block)
            _sparse_attn_sharedkv_decode_dynamic_cfa_dense_prefix_attention_kernel[
                (dense_grid,)
            ](
                q,
                compact_kv,
                compact_valid,
                cu_seqlens_q,
                seqused_kv,
                sinks,
                out,
                float(softmax_scale),
                TOTAL_Q=total_q,
                BATCH_SIZE=batch_size,
                BATCH_BLOCK=batch_block,
                NUM_Q_HEADS=64,
                HEAD_DIM=512,
                MAX_COMPACT_TOKENS=max_compact_tokens,
                HEAD_BLOCK=dense_head_block,
                CMP_BLOCK_N=64,
                HANDLE_ALIGNED=total_q > 1,
                **dynamic_options,
            )
        _LAST_GRAPHSAFE_WORKSPACE = (
            compact_kv,
            compact_valid,
        )
        return out

    if total_q == 1:
        return _launch_q1_scfa_direct_n64(
            q,
            compact_kv,
            compact_valid,
            sinks,
            cu_seqlens_q,
            seqused_kv,
            out,
            softmax_scale,
            batch_size,
            batch_block,
            dynamic_options,
        )
    # BiSheng生成错误结果，因此所有capacity统一保留5个chunk。
    chunk_num = 5
    chunk_tokens = 128
    # H32/N32 实测需要 2.135 Mbit UB，超过 910B 的 1.573 Mbit 上限；
    # H16/N32 保留跨 head 复用，同时让 masked fallback 稳定落入 UB。
    head_block = 16
    all_scfa_rows_short = eager_known_max_kv is not None and eager_known_max_kv <= 512
    direct_scfa_q1_q2 = total_q <= 2
    skip_scfa_partials = all_scfa_rows_short or direct_scfa_q1_q2
    partial_acc_shape = (total_q, chunk_num, 64, 512)
    partial_stat_shape = (total_q, chunk_num, 64)
    if skip_scfa_partials:
        # merge的short分支不读取partial指针；传稳定dummy地址即可，省掉两个
        # 必然no-op的partial launch和约total_q*5*64*512个FP32临时元素。
        partial_acc = _eager_workspace(
            "graphsafe.scfa.short_dummy_acc",
            (1,),
            dtype=torch.float32,
            device=q.device,
        )
        partial_max = partial_acc
        partial_sum = partial_acc
    elif eager_known_max_kv is not None:
        partial_acc = _eager_workspace(
            "graphsafe.scfa.partial_acc",
            partial_acc_shape,
            dtype=torch.float32,
            device=q.device,
        )
        partial_max = _eager_workspace(
            "graphsafe.scfa.partial_max",
            partial_stat_shape,
            dtype=torch.float32,
            device=q.device,
        )
        partial_sum = _eager_workspace(
            "graphsafe.scfa.partial_sum",
            partial_stat_shape,
            dtype=torch.float32,
            device=q.device,
        )
    else:
        partial_acc = torch.empty(
            partial_acc_shape, dtype=torch.float32, device=q.device
        )
        partial_max = torch.empty(
            partial_stat_shape, dtype=torch.float32, device=q.device
        )
        partial_sum = torch.empty_like(partial_max)
    fast_head_block = 32
    fast_partial_grid = total_q * triton.cdiv(64, fast_head_block) * chunk_num
    if not skip_scfa_partials:
        _sparse_attn_sharedkv_decode_dynamic_scfa_full_partial_kernel[
            (fast_partial_grid,)
        ](
            q,
            compact_kv,
            compact_valid,
            cu_seqlens_q,
            seqused_kv,
            sinks,
            partial_acc,
            partial_max,
            partial_sum,
            float(softmax_scale),
            TOTAL_Q=total_q,
            BATCH_SIZE=batch_size,
            BATCH_BLOCK=batch_block,
            NUM_Q_HEADS=64,
            HEAD_DIM=512,
            MAX_COMPACT_TOKENS=640,
            BLOCK_N=64,
            HEAD_BLOCK=fast_head_block,
            CHUNK_TOKENS=chunk_tokens,
            CHUNK_NUM=chunk_num,
        )
        partial_grid = total_q * triton.cdiv(64, head_block)
        _sparse_attn_sharedkv_decode_dynamic_scfa_partial_kernel[(partial_grid,)](
            q,
            compact_kv,
            compact_valid,
            cu_seqlens_q,
            seqused_kv,
            sinks,
            partial_acc,
            partial_max,
            partial_sum,
            float(softmax_scale),
            TOTAL_Q=total_q,
            BATCH_SIZE=batch_size,
            BATCH_BLOCK=batch_block,
            NUM_Q_HEADS=64,
            HEAD_DIM=512,
            MAX_COMPACT_TOKENS=640,
            BLOCK_N=32,
            HEAD_BLOCK=head_block,
            CHUNK_TOKENS=chunk_tokens,
            CHUNK_NUM=chunk_num,
        )
    # 动态长短混合继续用H8保持已验证的partial归并顺序。已知全短时merge
    # 直接完成QK/PV、不读取partial，H16可把program数减半且不改变归并树。
    merge_head_block = 8 if total_q == 1 else (16 if skip_scfa_partials else 8)
    merge_grid = total_q * triton.cdiv(64, merge_head_block)
    _sparse_attn_sharedkv_decode_dynamic_scfa_merge_kernel[(merge_grid,)](
        q,
        compact_kv,
        compact_valid,
        sinks,
        partial_acc,
        partial_max,
        partial_sum,
        cu_seqlens_q,
        seqused_kv,
        out,
        float(softmax_scale),
        TOTAL_Q=total_q,
        BATCH_SIZE=batch_size,
        BATCH_BLOCK=batch_block,
        NUM_Q_HEADS=64,
        HEAD_DIM=512,
        MAX_COMPACT_TOKENS=640,
        BLOCK_N=64,
        CHUNK_NUM=chunk_num,
        HEAD_BLOCK=merge_head_block,
        SHORT_COMPACT_LIMIT=256,
        DIRECT_ALL=direct_scfa_q1_q2,
        **dynamic_options,
    )
    _LAST_GRAPHSAFE_WORKSPACE = (
        compact_kv,
        compact_valid,
        partial_acc,
        partial_max,
        partial_sum,
    )
    return out


def sparse_attn_sharedkv_graphsafe_impl(
    q: torch.Tensor,
    *,
    ori_kv: torch.Tensor,
    cmp_kv: Optional[torch.Tensor] = None,
    ori_sparse_indices: Optional[torch.Tensor] = None,
    cmp_sparse_indices: Optional[torch.Tensor] = None,
    ori_block_table: torch.Tensor,
    cmp_block_table: Optional[torch.Tensor] = None,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_ori_kv: Optional[torch.Tensor] = None,
    cu_seqlens_cmp_kv: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_kv: torch.Tensor,
    sinks: torch.Tensor,
    metadata: Optional[torch.Tensor] = None,
    softmax_scale: float = 0,
    cmp_ratio: int = 0,
    ori_mask_mode: int = 4,
    cmp_mask_mode: int = 3,
    ori_win_left: int = 127,
    ori_win_right: int = 0,
    layout_q: str = "TND",
    layout_kv: str = "PA_ND",
    return_softmax_lse: bool = False,
    max_model_len: Optional[int] = None,
    eager_known_max_kv: Optional[int] = None,
) -> torch.Tensor:
    """Launch the dynamic graph-safe path without reading device values on host."""
    del cu_seqlens_ori_kv, cu_seqlens_cmp_kv, seqused_q, metadata
    if return_softmax_lse:
        raise NotImplementedError("softmax_lse is not supported.")
    if layout_q != "TND" or layout_kv != "PA_ND":
        raise NotImplementedError("The graph-safe path requires TND Q and PA_ND KV.")
    if q.dim() != 3 or tuple(q.shape[1:]) != (64, 512):
        raise ValueError(f"q must have shape [T, 64, 512], got {tuple(q.shape)}.")
    if q.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"q must be FP16 or BF16, got {q.dtype}.")
    if (
        ori_mask_mode != 4
        or cmp_mask_mode != 3
        or ori_win_left != 127
        or ori_win_right != 0
    ):
        raise ValueError(
            "Only ori/cmp mask modes 4/3 and window [127, 0] are supported."
        )
    if ori_sparse_indices is not None and ori_sparse_indices.numel() > 0:
        raise ValueError("ori_sparse_indices is not supported.")
    if ori_kv.dim() != 4 or tuple(ori_kv.shape[1:]) != (128, 1, 512):
        raise ValueError("ori_kv must have PA_ND shape [P, 128, 1, 512].")
    if ori_block_table.dim() != 2 or ori_block_table.dtype != torch.int32:
        raise ValueError("ori_block_table must be a rank-2 int32 tensor.")
    if cu_seqlens_q.dim() != 1 or cu_seqlens_q.dtype != torch.int32:
        raise ValueError("cu_seqlens_q must be a rank-1 int32 tensor.")
    if seqused_kv.dim() != 1 or seqused_kv.dtype != torch.int32:
        raise ValueError("seqused_kv must be a rank-1 int32 tensor.")
    batch_size = int(cu_seqlens_q.numel() - 1)
    if (
        batch_size <= 0
        or int(ori_block_table.shape[0]) != batch_size
        or seqused_kv.numel() != batch_size
    ):
        raise ValueError(
            "Batch dimensions of cu_seqlens_q, seqused_kv and ori_block_table must agree."
        )
    if sinks.dtype != torch.float32 or tuple(sinks.shape) != (64,):
        raise ValueError("sinks must have shape [64] and dtype float32.")

    has_cmp = isinstance(cmp_kv, torch.Tensor) and cmp_kv.numel() > 0
    has_sparse = (
        isinstance(cmp_sparse_indices, torch.Tensor) and cmp_sparse_indices.numel() > 0
    )
    if not has_cmp:
        if has_sparse or cmp_ratio not in (0, 1):
            raise ValueError("SWA requires no compressed inputs and cmp_ratio 0 or 1.")
        mode = 0
        cmp_ratio = 1
        cmp_kv = ori_kv
        cmp_block_table = ori_block_table
        cmp_sparse_indices = ori_block_table
    else:
        if cmp_kv.dim() != 4 or tuple(cmp_kv.shape[1:]) != (128, 1, 512):
            raise ValueError("cmp_kv must have PA_ND shape [P, 128, 1, 512].")
        if (
            cmp_block_table is None
            or cmp_block_table.dim() != 2
            or cmp_block_table.dtype != torch.int32
        ):
            raise ValueError("cmp_block_table must be a rank-2 int32 tensor.")
        if int(cmp_block_table.shape[0]) != batch_size:
            raise ValueError(
                "cmp_block_table batch dimension does not match cu_seqlens_q."
            )
        if has_sparse:
            if cmp_ratio != 4 or cmp_sparse_indices.dtype != torch.int32:
                raise ValueError("SCFA requires cmp_ratio=4 and int32 sparse indices.")
            if cmp_sparse_indices.dim() != 3 or tuple(cmp_sparse_indices.shape[1:]) != (
                1,
                512,
            ):
                raise ValueError("cmp_sparse_indices must have shape [T, 1, 512].")
            mode = 2
        else:
            if cmp_ratio != 128:
                raise ValueError("CFA requires cmp_ratio=128.")
            mode = 1
            cmp_sparse_indices = cmp_block_table

    if softmax_scale in (None, 0):
        softmax_scale = 512**-0.5
    return _launch_graphsafe_dynamic_compact_decode(
        q,
        ori_kv=ori_kv,
        cmp_kv=cmp_kv,
        cmp_sparse_indices=cmp_sparse_indices,
        ori_block_table=ori_block_table,
        cmp_block_table=cmp_block_table,
        cu_seqlens_q=cu_seqlens_q,
        seqused_kv=seqused_kv,
        sinks=sinks,
        softmax_scale=float(softmax_scale),
        cmp_ratio=int(cmp_ratio),
        mode=mode,
        batch_size=batch_size,
        max_model_len=max_model_len,
        eager_known_max_kv=eager_known_max_kv,
    )


@libentry()
@triton.jit
def _sparse_attn_sharedkv_decode_swa_pack_window_kv_kernel(
    ori_kv_ptr,
    ori_block_table_ptr,
    compact_kv_ptr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ORI_BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # DSV4 swa_decode compact-dot 的前置 pack：
    # SWA decode 只看最近 128 个 ori token。当前 testcase 中 kv_len=8193，
    # 窗口 token 为 [8065, 8193)，对应 page63 offset 1..127 与
    # page64 offset 0。先把这 128 个 token 搬成连续 [128, D]，主 kernel
    # 就可以完全避开 PageAttention 的不规则寻址。
    #
    # 这里做的是 PageAttention -> compact KV 的地址转换：
    #   逻辑 token id -> 逻辑 page id / page 内 offset
    #   逻辑 page id -> block_table 查物理 page id
    #   物理 page id + page offset -> PA_ND 中真实 K/V 地址
    # 因为 DSV4 的 decode case 固定只跨 page63/page64，所以这里把通用
    # page loop 特化成两个 page 的简单选择，减少主 attention kernel 的
    # 地址计算和分支。
    tl.static_assert(HEAD_DIM == 512, "SWA compact pack expects D=512.")
    tl.static_assert(BLOCK_D == 512, "SWA compact pack expects BLOCK_D=512.")
    tl.static_assert(ORI_BLOCK_SIZE == 128, "SWA compact pack expects ori block=128.")
    tl.static_assert(BLOCK_N == 16, "SWA compact pack expects BLOCK_N=16.")

    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)
    num_tiles = tl.cdiv(128, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    for tile_id in range(pid, num_tiles, n_programs):
        token = tile_id.to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
        block63 = tl.load(ori_block_table_ptr + 63)
        block64 = tl.load(ori_block_table_ptr + 64)
        logical_offset = token + 1
        physical_block = tl.where(logical_offset < ORI_BLOCK_SIZE, block63, block64)
        page_offset = tl.where(logical_offset < ORI_BLOCK_SIZE, logical_offset, 0)
        src = tl.load(
            ori_kv_ptr
            + (physical_block[:, None] * ORI_BLOCK_SIZE + page_offset[:, None])
            * HEAD_DIM
            + offs_d[None, :],
        )
        tl.store(compact_kv_ptr + token[:, None] * HEAD_DIM + offs_d[None, :], src)


@libentry()
@triton.jit
def _sparse_attn_sharedkv_decode_swa_compact_attention_kernel(
    q_ptr,
    compact_kv_ptr,
    sinks_ptr,
    out_ptr,
    softmax_scale,
    NUM_Q_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    TOTAL_KV: tl.constexpr,
):
    # SWA compact-dot 主 kernel：
    # 一个 program 处理 HEAD_BLOCK 个 query head，完整覆盖 D=512。相比旧
    # two-head vector 路径，这里让 8 个 head 共享同一份连续 K/V tile，并
    # 用 tl.dot 完成 QK 与 PV，避免 32 个 head-pair 重复读取同一段 shared KV。
    #
    # kernel 内部路线：
    # 1. 读取一组 query head 的 Q[HEAD_BLOCK, D]；
    # 2. sink 作为 softmax 里的额外常量 logit，初始化 row_max/row_sum；
    # 3. 按 64-token tile 扫 compact KV，共两轮覆盖 128 个窗口 token；
    # 4. 每轮做 QK、online softmax 合并、PV 累加；
    # 5. 最后除以 row_sum 写回 O[HEAD_BLOCK, D]。
    tl.static_assert(HEAD_DIM == 512, "SWA compact dot expects D=512.")
    tl.static_assert(BLOCK_D == 512, "SWA compact dot expects BLOCK_D=512.")
    tl.static_assert(BLOCK_N == 64, "SWA compact dot expects BLOCK_N=64.")
    tl.static_assert(TOTAL_KV == 128, "SWA compact dot expects 128 KV tokens.")

    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)
    head_blocks = tl.cdiv(NUM_Q_HEADS, HEAD_BLOCK)
    offs_d = tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)

    for head_block_id in range(pid, head_blocks, n_programs):
        head_base = head_block_id.to(tl.int64) * HEAD_BLOCK
        offs_h = head_base + tl.arange(0, HEAD_BLOCK).to(tl.int64)
        valid_h = offs_h < NUM_Q_HEADS

        q = tl.load(
            q_ptr + offs_h[:, None] * HEAD_DIM + offs_d[None, :],
            mask=valid_h[:, None],
            other=0.0,
        )
        row_max = tl.load(sinks_ptr + offs_h, mask=valid_h, other=0.0).to(tl.float32)
        row_sum = tl.full((HEAD_BLOCK,), 1.0, tl.float32)
        acc = tl.zeros((HEAD_BLOCK, BLOCK_D), dtype=tl.float32)

        for token_base in tl.static_range(0, TOTAL_KV, BLOCK_N):
            token = token_base + offs_n
            kv = tl.load(compact_kv_ptr + token[:, None] * HEAD_DIM + offs_d[None, :])
            scores = tl.dot(q, tl.trans(kv)) * softmax_scale
            scores = tl.where(valid_h[:, None], scores, float("-inf"))

            new_max = tl.maximum(row_max, tl.max(scores, axis=1))
            p = tl.exp(scores - new_max[:, None])
            p = tl.where(valid_h[:, None], p, 0.0)
            alpha = tl.exp(row_max - new_max)
            row_sum = row_sum * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]

            acc += tl.dot(p.to(kv.dtype), kv)
            row_max = new_max

        acc = acc / row_sum[:, None]
        tl.store(
            out_ptr + offs_h[:, None] * HEAD_DIM + offs_d[None, :],
            acc.to(out_ptr.dtype.element_ty),
            mask=valid_h[:, None],
        )


@libentry()
@triton.jit
def _sparse_attn_sharedkv_decode_cfa_pack_window_and_cmp_kv_kernel(
    ori_kv_ptr,
    cmp_kv_ptr,
    ori_block_table_ptr,
    cmp_block_table_ptr,
    compact_kv_ptr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ORI_BLOCK_SIZE: tl.constexpr,
    CMP_BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # DSV4 cfa_decode compact-dot 专用的前置 pack：
    # 把 PA_ND 下实际可见的 128 个 ori token + 64 个 cmp token 拷贝成
    # 连续 [192, D]。主计算 kernel 就能用规整地址做 tl.dot，避免把 PA
    # gather 和 cube dot 混在同一个 kernel 里。
    tl.static_assert(HEAD_DIM == 512, "CFA compact pack expects D=512.")
    tl.static_assert(BLOCK_D == 512, "CFA compact pack expects BLOCK_D=512.")
    tl.static_assert(ORI_BLOCK_SIZE == 128, "CFA compact pack expects ori block=128.")
    tl.static_assert(CMP_BLOCK_SIZE == 128, "CFA compact pack expects cmp block=128.")
    tl.static_assert(BLOCK_N == 16, "CFA compact pack expects BLOCK_N=16.")

    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)
    num_tiles = tl.cdiv(192, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    for tile_id in range(pid, num_tiles, n_programs):
        token = tile_id.to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
        is_ori = token < 128

        ori_block63 = tl.load(ori_block_table_ptr + 63)
        ori_block64 = tl.load(ori_block_table_ptr + 64)
        cmp_block0 = tl.load(cmp_block_table_ptr)

        ori_logical_offset = token + 1
        ori_physical_block = tl.where(
            ori_logical_offset < ORI_BLOCK_SIZE, ori_block63, ori_block64
        )
        ori_page_offset = tl.where(
            ori_logical_offset < ORI_BLOCK_SIZE, ori_logical_offset, 0
        )
        cmp_pos = token - 128
        safe_cmp_pos = tl.maximum(cmp_pos, 0)

        src = tl.load(
            ori_kv_ptr
            + (ori_physical_block[:, None] * ORI_BLOCK_SIZE + ori_page_offset[:, None])
            * HEAD_DIM
            + offs_d[None, :],
            mask=is_ori[:, None],
            other=0.0,
        )
        src_cmp = tl.load(
            cmp_kv_ptr
            + (cmp_block0 * CMP_BLOCK_SIZE + safe_cmp_pos[:, None]) * HEAD_DIM
            + offs_d[None, :],
            mask=(~is_ori)[:, None],
            other=0.0,
        )
        src = tl.where(is_ori[:, None], src, src_cmp)
        tl.store(compact_kv_ptr + token[:, None] * HEAD_DIM + offs_d[None, :], src)


@libentry()
@triton.jit
def _sparse_attn_sharedkv_decode_cfa_compact_attention_kernel(
    q_ptr,
    compact_kv_ptr,
    sinks_ptr,
    out_ptr,
    softmax_scale,
    NUM_Q_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    TOTAL_KV: tl.constexpr,
):
    # compact-dot 专用主 kernel：
    # 一个 program 处理 HEAD_BLOCK 个 query head。所有 head 共享同一份
    # 连续 compact_kv tile，通过 tl.dot 做 QK 和 PV，验证 cube 路径在
    # 去掉 PA gather 后是否能显著降低当前 partial 的重复 K/V 读取。
    tl.static_assert(HEAD_DIM == 512, "CFA compact dot expects D=512.")
    tl.static_assert(BLOCK_D == 512, "CFA compact dot expects BLOCK_D=512.")
    tl.static_assert(BLOCK_N == 64, "CFA compact dot expects BLOCK_N=64.")
    tl.static_assert(TOTAL_KV == 192, "CFA compact dot expects 192 KV tokens.")

    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)
    head_blocks = tl.cdiv(NUM_Q_HEADS, HEAD_BLOCK)
    offs_d = tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)

    for head_block_id in range(pid, head_blocks, n_programs):
        head_base = head_block_id.to(tl.int64) * HEAD_BLOCK
        offs_h = head_base + tl.arange(0, HEAD_BLOCK).to(tl.int64)
        valid_h = offs_h < NUM_Q_HEADS

        q = tl.load(
            q_ptr + offs_h[:, None] * HEAD_DIM + offs_d[None, :],
            mask=valid_h[:, None],
            other=0.0,
        )
        row_max = tl.load(sinks_ptr + offs_h, mask=valid_h, other=0.0).to(tl.float32)
        row_sum = tl.full((HEAD_BLOCK,), 1.0, tl.float32)
        acc = tl.zeros((HEAD_BLOCK, BLOCK_D), dtype=tl.float32)

        for token_base in tl.range(0, TOTAL_KV, BLOCK_N):
            token = token_base + offs_n
            kv = tl.load(compact_kv_ptr + token[:, None] * HEAD_DIM + offs_d[None, :])
            scores = tl.dot(q, tl.trans(kv)) * softmax_scale
            scores = tl.where(valid_h[:, None], scores, float("-inf"))
            new_max = tl.maximum(row_max, tl.max(scores, axis=1))
            p = tl.exp(scores - new_max[:, None])
            p = tl.where(valid_h[:, None], p, 0.0)
            alpha = tl.exp(row_max - new_max)
            row_sum = row_sum * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]

            acc += tl.dot(p.to(kv.dtype), kv)
            row_max = new_max

        acc = acc / row_sum[:, None]
        tl.store(
            out_ptr + offs_h[:, None] * HEAD_DIM + offs_d[None, :],
            acc.to(out_ptr.dtype.element_ty),
            mask=valid_h[:, None],
        )


@libentry()
@triton.jit
def _sparse_attn_sharedkv_decode_scfa_pack_window_and_sparse_kv_kernel(
    ori_kv_ptr,
    cmp_kv_ptr,
    cmp_sparse_indices_ptr,
    ori_block_table_ptr,
    cmp_block_table_ptr,
    compact_kv_ptr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ORI_BLOCK_SIZE: tl.constexpr,
    CMP_BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CMP_THRESHOLD: tl.constexpr,
    TOTAL_KV: tl.constexpr,
    ORI_TOKENS: tl.constexpr,
):
    # DSV4 scfa_decode compact-dot 的前置 pack：
    # compact token 0..127 来自最近 128 个 ori token；128..639 来自
    # cmp_sparse_indices 中的 512 个 compressed token。这样主 attention
    # kernel 可以完全摆脱 PA page table 和 sparse index gather。
    tl.static_assert(HEAD_DIM == 512, "SCFA compact pack expects D=512.")
    tl.static_assert(BLOCK_D == 512, "SCFA compact pack expects BLOCK_D=512.")
    tl.static_assert(ORI_BLOCK_SIZE == 128, "SCFA compact pack expects ori block=128.")
    tl.static_assert(CMP_BLOCK_SIZE == 128, "SCFA compact pack expects cmp block=128.")
    tl.static_assert(BLOCK_N == 16, "SCFA compact pack expects BLOCK_N=16.")
    tl.static_assert(TOTAL_KV == 640, "SCFA compact pack expects 640 tokens.")
    tl.static_assert(ORI_TOKENS == 128, "SCFA compact pack expects 128 ori tokens.")
    tl.static_assert(
        CMP_THRESHOLD == 2048, "SCFA compact pack expects cmp_threshold=2048."
    )

    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)
    num_tiles = tl.cdiv(TOTAL_KV, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    for tile_id in range(pid, num_tiles, n_programs):
        token = tile_id.to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
        is_ori = token < ORI_TOKENS

        ori_block63 = tl.load(ori_block_table_ptr + 63)
        ori_block64 = tl.load(ori_block_table_ptr + 64)
        ori_logical_offset = token + 1
        ori_physical_block = tl.where(
            ori_logical_offset < ORI_BLOCK_SIZE, ori_block63, ori_block64
        )
        ori_page_offset = tl.where(
            ori_logical_offset < ORI_BLOCK_SIZE, ori_logical_offset, 0
        )
        ori_src = tl.load(
            ori_kv_ptr
            + (ori_physical_block[:, None] * ORI_BLOCK_SIZE + ori_page_offset[:, None])
            * HEAD_DIM
            + offs_d[None, :],
            mask=is_ori[:, None],
            other=0.0,
        )

        topk_id = token - ORI_TOKENS
        sparse_idx = tl.load(
            cmp_sparse_indices_ptr + topk_id,
            mask=(~is_ori) & (topk_id < 512),
            other=-1,
        ).to(tl.int64)
        valid_sparse = (sparse_idx >= 0) & (sparse_idx < CMP_THRESHOLD)
        safe_sparse_idx = tl.maximum(sparse_idx, 0)
        cmp_page_id = safe_sparse_idx // CMP_BLOCK_SIZE
        cmp_page_offset = safe_sparse_idx - cmp_page_id * CMP_BLOCK_SIZE
        cmp_physical_block = tl.load(
            cmp_block_table_ptr + cmp_page_id, mask=valid_sparse, other=0
        )
        cmp_src = tl.load(
            cmp_kv_ptr
            + (cmp_physical_block[:, None] * CMP_BLOCK_SIZE + cmp_page_offset[:, None])
            * HEAD_DIM
            + offs_d[None, :],
            mask=valid_sparse[:, None],
            other=0.0,
        )
        src = tl.where(is_ori[:, None], ori_src, cmp_src)
        tl.store(
            compact_kv_ptr + token[:, None] * HEAD_DIM + offs_d[None, :],
            src,
            mask=token[:, None] < TOTAL_KV,
        )


@libentry()
@triton.jit
def _sparse_attn_sharedkv_decode_scfa_sparse_chunk_attention_kernel(
    q_ptr,
    compact_kv_ptr,
    sinks_ptr,
    partial_acc_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    softmax_scale,
    NUM_Q_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    CHUNK_TOKENS: tl.constexpr,
    CHUNK_NUM: tl.constexpr,
    SINK_IN_PARTIAL: tl.constexpr,
):
    # SCFA compact-dot token-split partial：
    # compact_kv 已经是连续 [640, D]，这里把 token 轴切成多个 chunk，
    # 一个 program 处理 HEAD_BLOCK 个 head 和一个 token chunk。这样 640
    # token 的 QK/PV 不再由单个 program 串行扫完，而是交给更多 AI core
    # 并行执行；随后复用已有 reduce kernel 合并 online-softmax 状态。
    tl.static_assert(HEAD_DIM == 512, "SCFA compact partial expects D=512.")
    tl.static_assert(BLOCK_D == 512, "SCFA compact partial expects BLOCK_D=512.")
    tl.static_assert(BLOCK_N == 64, "SCFA compact partial expects BLOCK_N=64.")

    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)
    head_blocks = tl.cdiv(NUM_Q_HEADS, HEAD_BLOCK)
    num_tasks = head_blocks * CHUNK_NUM
    offs_d = tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)

    for task_id in range(pid, num_tasks, n_programs):
        head_block_id = (task_id // CHUNK_NUM).to(tl.int64)
        chunk_id = (task_id - head_block_id * CHUNK_NUM).to(tl.int64)
        head_base = head_block_id * HEAD_BLOCK
        offs_h = head_base + tl.arange(0, HEAD_BLOCK).to(tl.int64)
        valid_h = offs_h < NUM_Q_HEADS

        q = tl.load(
            q_ptr + offs_h[:, None] * HEAD_DIM + offs_d[None, :],
            mask=valid_h[:, None],
            other=0.0,
        )
        if SINK_IN_PARTIAL:
            if chunk_id == 0:
                row_max = tl.load(sinks_ptr + offs_h, mask=valid_h, other=0.0).to(
                    tl.float32
                )
                row_sum = tl.full((HEAD_BLOCK,), 1.0, tl.float32)
            else:
                row_max = tl.full((HEAD_BLOCK,), float("-inf"), tl.float32)
                row_sum = tl.full((HEAD_BLOCK,), 0.0, tl.float32)
        else:
            row_max = tl.full((HEAD_BLOCK,), float("-inf"), tl.float32)
            row_sum = tl.full((HEAD_BLOCK,), 0.0, tl.float32)
        acc = tl.zeros((HEAD_BLOCK, BLOCK_D), dtype=tl.float32)

        chunk_base = chunk_id * CHUNK_TOKENS
        for local_base in tl.range(0, CHUNK_TOKENS, BLOCK_N):
            token = chunk_base + local_base + offs_n
            kv = tl.load(compact_kv_ptr + token[:, None] * HEAD_DIM + offs_d[None, :])
            scores = tl.dot(q, tl.trans(kv)) * softmax_scale
            scores = tl.where(valid_h[:, None], scores, float("-inf"))
            new_max = tl.maximum(row_max, tl.max(scores, axis=1))
            p = tl.exp(scores - new_max[:, None])
            p = tl.where(valid_h[:, None], p, 0.0)
            alpha = tl.exp(row_max - new_max)
            row_sum = row_sum * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]

            acc += tl.dot(p.to(kv.dtype), kv)
            row_max = new_max

        partial_base = chunk_id * NUM_Q_HEADS + offs_h
        tl.store(partial_max_ptr + partial_base, row_max, mask=valid_h)
        tl.store(partial_sum_ptr + partial_base, row_sum, mask=valid_h)
        tl.store(
            partial_acc_ptr + partial_base[:, None] * HEAD_DIM + offs_d[None, :],
            acc,
            mask=valid_h[:, None],
        )


@libentry()
@triton.jit
def _sparse_attn_sharedkv_decode_scfa_merge_chunks_kernel(
    partial_acc_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    sinks_ptr,
    out_ptr,
    NUM_Q_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    CHUNK_NUM: tl.constexpr,
    SINK_IN_PARTIAL: tl.constexpr,
):
    # SCFA decode token-split 第 2 段：
    # 合并若干个 chunk 的局部 online-softmax 状态。这个 reduce 比 CFA
    # 更重，但能换来 partial 阶段把完整 topK 串行长任务拆成更多短任务。
    tl.static_assert(HEAD_DIM == 512, "SCFA token-split reduce expects D=512.")
    tl.static_assert(BLOCK_D == 512, "SCFA token-split reduce expects BLOCK_D=512.")

    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)
    num_pairs = tl.cdiv(NUM_Q_HEADS, 2)
    offs_d = tl.arange(0, BLOCK_D)

    for pair_id in range(pid, num_pairs, n_programs):
        head0 = pair_id.to(tl.int64) * 2
        head1 = head0 + 1

        row_max0 = tl.full((), float("-inf"), tl.float32)
        row_max1 = tl.full((), float("-inf"), tl.float32)
        for chunk in tl.static_range(0, CHUNK_NUM):
            row_max0 = tl.maximum(
                row_max0, tl.load(partial_max_ptr + chunk * NUM_Q_HEADS + head0)
            )
            row_max1 = tl.maximum(
                row_max1, tl.load(partial_max_ptr + chunk * NUM_Q_HEADS + head1)
            )
        if not SINK_IN_PARTIAL:
            sink0 = tl.load(sinks_ptr + head0).to(tl.float32)
            sink1 = tl.load(sinks_ptr + head1).to(tl.float32)
            row_max0 = tl.maximum(row_max0, sink0)
            row_max1 = tl.maximum(row_max1, sink1)

        row_sum0 = tl.full((), 0.0, tl.float32)
        row_sum1 = tl.full((), 0.0, tl.float32)
        for chunk in tl.static_range(0, CHUNK_NUM):
            max0 = tl.load(partial_max_ptr + chunk * NUM_Q_HEADS + head0)
            max1 = tl.load(partial_max_ptr + chunk * NUM_Q_HEADS + head1)
            sum0 = tl.load(partial_sum_ptr + chunk * NUM_Q_HEADS + head0)
            sum1 = tl.load(partial_sum_ptr + chunk * NUM_Q_HEADS + head1)
            row_sum0 += sum0 * tl.exp(max0 - row_max0)
            row_sum1 += sum1 * tl.exp(max1 - row_max1)
        if not SINK_IN_PARTIAL:
            row_sum0 += tl.exp(sink0 - row_max0)
            row_sum1 += tl.exp(sink1 - row_max1)

        inv_sum0 = 1.0 / row_sum0
        inv_sum1 = 1.0 / row_sum1
        acc0 = tl.zeros((BLOCK_D,), dtype=tl.float32)
        acc1 = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for chunk in tl.static_range(0, CHUNK_NUM):
            max0 = tl.load(partial_max_ptr + chunk * NUM_Q_HEADS + head0)
            max1 = tl.load(partial_max_ptr + chunk * NUM_Q_HEADS + head1)
            scale0 = tl.exp(max0 - row_max0) * inv_sum0
            scale1 = tl.exp(max1 - row_max1) * inv_sum1
            acc0 += (
                tl.load(
                    partial_acc_ptr + (chunk * NUM_Q_HEADS + head0) * HEAD_DIM + offs_d
                ).to(tl.float32)
                * scale0
            )
            acc1 += (
                tl.load(
                    partial_acc_ptr + (chunk * NUM_Q_HEADS + head1) * HEAD_DIM + offs_d
                ).to(tl.float32)
                * scale1
            )

        tl.store(out_ptr + head0 * HEAD_DIM + offs_d, acc0.to(out_ptr.dtype.element_ty))
        tl.store(out_ptr + head1 * HEAD_DIM + offs_d, acc1.to(out_ptr.dtype.element_ty))


def _flatten_q(
    q: torch.Tensor,
    layout_q: str,
    cu_seqlens_q: Optional[torch.Tensor],
) -> tuple[torch.Tensor, tuple[int, int, int, int, int], bool]:
    if layout_q == "TND":
        if q.dim() != 3:
            raise ValueError(
                f"layout_q=TND expects q shape [T, N, D], got {tuple(q.shape)}."
            )
        total_q, num_heads_q, head_dim = q.shape
        if cu_seqlens_q is None:
            raise ValueError("layout_q=TND requires cu_seqlens_q.")
        batch_size = int(cu_seqlens_q.numel() - 1)
        cu_cpu = cu_seqlens_q.detach().cpu()
        max_q_len = int((cu_cpu[1:] - cu_cpu[:-1]).max().item())
        return (
            q.contiguous(),
            (batch_size, max_q_len, total_q, num_heads_q, head_dim),
            True,
        )

    if layout_q == "BSND":
        if q.dim() != 4:
            raise ValueError(
                f"layout_q=BSND expects q shape [B, S, N, D], got {tuple(q.shape)}."
            )
        batch_size, max_q_len, num_heads_q, head_dim = q.shape
        q_flat = q.contiguous().view(batch_size * max_q_len, num_heads_q, head_dim)
        return (
            q_flat,
            (batch_size, max_q_len, batch_size * max_q_len, num_heads_q, head_dim),
            False,
        )

    raise ValueError(f"Unsupported layout_q={layout_q}. Expected 'BSND' or 'TND'.")


def _try_dsv4_decode_compact_attention(
    q: torch.Tensor,
    *,
    ori_kv: torch.Tensor,
    cmp_kv: Optional[torch.Tensor],
    cmp_sparse_indices: Optional[torch.Tensor],
    ori_block_table: torch.Tensor,
    cmp_block_table: Optional[torch.Tensor],
    cu_seqlens_q: torch.Tensor,
    seqused_kv: torch.Tensor,
    sinks: Optional[torch.Tensor],
    softmax_scale: float,
    cmp_ratio: int,
    ori_win_left: int,
    ori_win_right: int,
    layout_q: str,
    layout_kv: str,
) -> Optional[torch.Tensor]:
    """DSV4 decode 专用快路径。

    这个路径现在合并在主实现中默认生效。
    它只覆盖当前已验证的三个 decode 形状。核心思路是先把分页 KV/cache
    和 sparse topK gather 结果 pack 成连续的 `compact_kv`，再让多个 query
    head 共用这份连续 K/V tile 做 `tl.dot`。这样主计算 kernel 不再同时处理
    PageAttention 的不规则寻址和 attention dot，decode 场景能显著减少重复
    K/V 读取。
    """
    if layout_q != "TND" or layout_kv != "PA_ND":
        return None
    if (
        q.dim() != 3
        or int(q.shape[0]) != 1
        or int(q.shape[1]) != 64
        or int(q.shape[2]) != 512
    ):
        return None
    if cu_seqlens_q is None or int(cu_seqlens_q.numel()) != 2:
        return None
    cu_cpu = cu_seqlens_q.detach().cpu()
    if int(cu_cpu[0].item()) != 0 or int(cu_cpu[1].item()) != 1:
        return None
    if int(ori_win_left) != 127 or int(ori_win_right) != 0:
        return None
    if (
        int(ori_kv.shape[1]) != 128
        or int(ori_kv.shape[2]) != 1
        or int(ori_kv.shape[3]) != 512
    ):
        return None
    if seqused_kv is None or int(seqused_kv.numel()) == 0:
        return None
    kv_len = int(seqused_kv.detach().cpu()[0].item())
    if kv_len != 8193:
        return None

    q = q.contiguous()
    ori_kv = ori_kv.contiguous()
    ori_block_table = ori_block_table.contiguous()
    seqused_kv = seqused_kv.contiguous()
    num_q_heads = int(q.shape[1])
    head_dim = int(q.shape[2])
    block_d = triton.next_power_of_2(head_dim)
    out = torch.empty_like(q)
    if softmax_scale in (None, 0):
        softmax_scale = head_dim**-0.5
    if sinks is None or sinks.numel() == 0:
        sinks = torch.zeros((num_q_heads,), dtype=torch.float32, device=q.device)
    else:
        sinks = sinks.contiguous()
    decode_worker_grid = get_num_cores("cube")

    has_cmp_kv = cmp_kv is not None and cmp_kv.numel() > 0
    has_cmp_sparse_indices = (
        cmp_sparse_indices is not None and cmp_sparse_indices.numel() > 0
    )

    if not has_cmp_kv:
        # swa_decode 当前 kernel 路线：
        #   kernel1 pack_window_kv:
        #     从 PA_ND ori_kv + ori_block_table 中取最近 128 个原始 KV，
        #     pack 成连续 compact_kv[128, D]。
        #   kernel2 compact_attention:
        #     多个 query head 共享 compact_kv，用 tl.dot 完成
        #     QK -> online softmax(+sink) -> PV -> O。
        #
        # 这个 decode case 不需要 compressed KV，也不需要 sparse topK；
        # 因此 `has_cmp_kv=False` 时只走 original SWA window。
        total_kv = 128
        compact_kv = torch.empty(
            (total_kv, head_dim), dtype=ori_kv.dtype, device=q.device
        )
        _sparse_attn_sharedkv_decode_swa_pack_window_kv_kernel[(decode_worker_grid,)](
            ori_kv,
            ori_block_table,
            compact_kv,
            HEAD_DIM=head_dim,
            BLOCK_D=block_d,
            ORI_BLOCK_SIZE=int(ori_kv.shape[1]),
            BLOCK_N=16,
        )
        _sparse_attn_sharedkv_decode_swa_compact_attention_kernel[
            (decode_worker_grid,)
        ](
            q,
            compact_kv,
            sinks,
            out,
            float(softmax_scale),
            NUM_Q_HEADS=num_q_heads,
            HEAD_DIM=head_dim,
            BLOCK_D=block_d,
            BLOCK_N=64,
            HEAD_BLOCK=16,
            TOTAL_KV=total_kv,
        )
        _retain_eager_workspace(compact_kv)
        return out

    cmp_kv = cmp_kv.contiguous()
    if cmp_block_table is None:
        return None
    cmp_block_table = cmp_block_table.contiguous()
    if (
        int(cmp_kv.shape[1]) != 128
        or int(cmp_kv.shape[2]) != 1
        or int(cmp_kv.shape[3]) != head_dim
    ):
        return None

    if (
        has_cmp_sparse_indices
        and int(cmp_ratio) == 4
        and int(cmp_sparse_indices.shape[-1]) == 512
    ):
        # SCFA decode：128 个原始 token + 512 个 sparse compressed token。
        # 先按 topK index pack 成 [640, D]，再把 token 轴切成两个 320-token
        # chunk 并行计算 partial，最后 reduce 合并 online-softmax 状态。
        cmp_sparse_indices = cmp_sparse_indices.contiguous()
        total_kv = 640
        cmp_threshold = kv_len // int(cmp_ratio)
        if cmp_threshold != 2048:
            return None
        compact_kv = torch.empty(
            (total_kv, head_dim), dtype=ori_kv.dtype, device=q.device
        )
        _sparse_attn_sharedkv_decode_scfa_pack_window_and_sparse_kv_kernel[
            (decode_worker_grid,)
        ](
            ori_kv,
            cmp_kv,
            cmp_sparse_indices,
            ori_block_table,
            cmp_block_table,
            compact_kv,
            HEAD_DIM=head_dim,
            BLOCK_D=block_d,
            ORI_BLOCK_SIZE=int(ori_kv.shape[1]),
            CMP_BLOCK_SIZE=int(cmp_kv.shape[1]),
            BLOCK_N=16,
            CMP_THRESHOLD=cmp_threshold,
            TOTAL_KV=total_kv,
            ORI_TOKENS=128,
        )
        head_block = 16
        # 640 tokens 切成 5 个 128-token chunk：相比旧 2 chunk，
        # partial attention 长任务明显缩短；相比 10 chunk，merge 开销不会
        # 反噬收益。该配置已在六个固定形状上验证。
        chunk_num = 5
        chunk_tokens = total_kv // chunk_num
        if chunk_tokens % 64 != 0:
            return None
        sink_in_partial = True
        partial_acc = torch.empty(
            (chunk_num, num_q_heads, head_dim), dtype=torch.float32, device=q.device
        )
        partial_max = torch.empty(
            (chunk_num, num_q_heads), dtype=torch.float32, device=q.device
        )
        partial_sum = torch.empty(
            (chunk_num, num_q_heads), dtype=torch.float32, device=q.device
        )
        _sparse_attn_sharedkv_decode_scfa_sparse_chunk_attention_kernel[
            (decode_worker_grid,)
        ](
            q,
            compact_kv,
            sinks,
            partial_acc,
            partial_max,
            partial_sum,
            float(softmax_scale),
            NUM_Q_HEADS=num_q_heads,
            HEAD_DIM=head_dim,
            BLOCK_D=block_d,
            BLOCK_N=64,
            HEAD_BLOCK=head_block,
            CHUNK_TOKENS=chunk_tokens,
            CHUNK_NUM=chunk_num,
            SINK_IN_PARTIAL=sink_in_partial,
        )
        _sparse_attn_sharedkv_decode_scfa_merge_chunks_kernel[(decode_worker_grid,)](
            partial_acc,
            partial_max,
            partial_sum,
            sinks,
            out,
            NUM_Q_HEADS=num_q_heads,
            HEAD_DIM=head_dim,
            BLOCK_D=block_d,
            CHUNK_NUM=chunk_num,
            SINK_IN_PARTIAL=sink_in_partial,
        )
        _retain_eager_workspace(compact_kv, partial_acc, partial_max, partial_sum)
        return out

    if (not has_cmp_sparse_indices) and int(cmp_ratio) == 128:
        # CFA decode：128 个原始 token + 64 个 compressed token。compressed
        # 部分是连续前缀，不需要 sparse index，因此 pack 后直接 compact-dot。
        total_kv = 192
        compact_kv = torch.empty(
            (total_kv, head_dim), dtype=ori_kv.dtype, device=q.device
        )
        _sparse_attn_sharedkv_decode_cfa_pack_window_and_cmp_kv_kernel[
            (decode_worker_grid,)
        ](
            ori_kv,
            cmp_kv,
            ori_block_table,
            cmp_block_table,
            compact_kv,
            HEAD_DIM=head_dim,
            BLOCK_D=block_d,
            ORI_BLOCK_SIZE=int(ori_kv.shape[1]),
            CMP_BLOCK_SIZE=int(cmp_kv.shape[1]),
            BLOCK_N=16,
        )
        _sparse_attn_sharedkv_decode_cfa_compact_attention_kernel[
            (decode_worker_grid,)
        ](
            q,
            compact_kv,
            sinks,
            out,
            float(softmax_scale),
            NUM_Q_HEADS=num_q_heads,
            HEAD_DIM=head_dim,
            BLOCK_D=block_d,
            BLOCK_N=64,
            HEAD_BLOCK=8,
            TOTAL_KV=total_kv,
        )
        _retain_eager_workspace(compact_kv)
        return out

    return None


def _pack_chunked_kv_range(
    paged_kv: torch.Tensor,
    block_table_row: torch.Tensor,
    *,
    logical_start: int,
    token_count: int,
    storage_count: Optional[int] = None,
    cache_tag: str,
    stream_key: tuple[int, int],
    explicit_stream: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
    """将单个序列的一段PA KV整理为连续K和FP16 V。"""
    if token_count < 0:
        raise ValueError("chunk pack token_count cannot be negative")
    storage_count = token_count if storage_count is None else storage_count
    if storage_count < token_count:
        raise ValueError("chunk pack storage_count cannot be smaller than token_count")
    shape = (storage_count, 512)
    logical_k = _eager_workspace(
        f"{cache_tag}.k",
        shape,
        dtype=paged_kv.dtype,
        device=paged_kv.device,
        stream_key=stream_key,
    )
    logical_v = _eager_workspace(
        f"{cache_tag}.v",
        shape,
        dtype=torch.float16,
        device=paged_kv.device,
        stream_key=stream_key,
    )
    grid = (get_num_cores("cube"),)
    constexpr_names = (
        "LOGICAL_START",
        "TOKEN_COUNT",
        "STORAGE_COUNT",
        "TABLE_WIDTH",
        "HEAD_DIM",
        "PAGE_SIZE",
        "BLOCK_N",
        "BLOCK_D",
    )
    constexpr_values = (
        int(logical_start),
        int(token_count),
        int(storage_count),
        int(block_table_row.numel()),
        512,
        128,
        64,
        256,
    )
    _launch_q256_cached_eager_kernel(
        stream_key if explicit_stream else None,
        _sparse_attn_sharedkv_chunk_pack_logical_k_and_fp16_v_kernel,
        grid,
        (logical_k, logical_v, paged_kv, block_table_row),
        constexpr_names,
        constexpr_values,
        _EAGER_BATCHED_PACK_OPTIONS,
        (paged_kv.dtype, *constexpr_values),
    )
    return logical_k, logical_v, (logical_k, logical_v)


def _pack_q256_ori_cmp_ranges(
    ori_kv: torch.Tensor,
    cmp_kv: torch.Tensor,
    ori_block_table_row: torch.Tensor,
    cmp_block_table_row: torch.Tensor,
    *,
    ori_logical_start: int,
    ori_token_count: int,
    ori_storage_count: int,
    cmp_token_count: int,
    cmp_storage_count: int,
    write_cmp_v: bool,
    cache_tag: str,
    stream_key: tuple[int, int],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[torch.Tensor, ...],
]:
    """Pack q256 original and compressed ranges with one device launch."""
    logical_ori_k = _eager_workspace(
        f"{cache_tag}.ori_k",
        (ori_storage_count, 512),
        dtype=ori_kv.dtype,
        device=ori_kv.device,
        stream_key=stream_key,
    )
    logical_ori_v = _eager_workspace(
        f"{cache_tag}.ori_v",
        (ori_storage_count, 512),
        dtype=torch.float16,
        device=ori_kv.device,
        stream_key=stream_key,
    )
    logical_cmp_k = _eager_workspace(
        f"{cache_tag}.cmp_k",
        (cmp_storage_count, 512),
        dtype=cmp_kv.dtype,
        device=cmp_kv.device,
        stream_key=stream_key,
    )
    if write_cmp_v:
        logical_cmp_v = _eager_workspace(
            f"{cache_tag}.cmp_v",
            (cmp_storage_count, 512),
            dtype=torch.float16,
            device=cmp_kv.device,
            stream_key=stream_key,
        )
        packed = (logical_ori_k, logical_ori_v, logical_cmp_k, logical_cmp_v)
    else:
        logical_cmp_v = logical_cmp_k
        packed = (logical_ori_k, logical_ori_v, logical_cmp_k)

    constexpr_names = (
        "ORI_LOGICAL_START",
        "ORI_TOKEN_COUNT",
        "ORI_STORAGE_COUNT",
        "CMP_TOKEN_COUNT",
        "CMP_STORAGE_COUNT",
        "ORI_TABLE_WIDTH",
        "CMP_TABLE_WIDTH",
        "HEAD_DIM",
        "PAGE_SIZE",
        "BLOCK_N",
        "BLOCK_D",
        "WRITE_CMP_V",
    )
    constexpr_values = (
        int(ori_logical_start),
        int(ori_token_count),
        int(ori_storage_count),
        int(cmp_token_count),
        int(cmp_storage_count),
        int(ori_block_table_row.numel()),
        int(cmp_block_table_row.numel()),
        512,
        128,
        64,
        256,
        bool(write_cmp_v),
    )
    _launch_q256_cached_eager_kernel(
        stream_key,
        _sparse_attn_sharedkv_q256_fused_ori_cmp_pack_kernel,
        (get_num_cores("cube"),),
        (
            logical_ori_k,
            logical_ori_v,
            logical_cmp_k,
            logical_cmp_v,
            ori_kv,
            cmp_kv,
            ori_block_table_row,
            cmp_block_table_row,
        ),
        constexpr_names,
        constexpr_values,
        _EAGER_BATCHED_PACK_OPTIONS,
        (
            ori_kv.dtype,
            cmp_kv.dtype,
            ori_block_table_row.dtype,
            cmp_block_table_row.dtype,
            *constexpr_values,
        ),
    )
    return logical_ori_k, logical_ori_v, logical_cmp_k, logical_cmp_v, packed


def _group_consecutive_q_lens(
    q_lens: list[int], max_group_size: int
) -> tuple[tuple[int, int, int, int], ...]:
    """Return positive-q runs as (batch_start, batch_end, q_start, q_len)."""
    if max_group_size <= 0:
        raise ValueError("max_group_size must be positive")
    groups: list[tuple[int, int, int, int]] = []
    q_start = 0
    batch_start = 0
    while batch_start < len(q_lens):
        q_count = q_lens[batch_start]
        if q_count < 0:
            raise ValueError("q_lens cannot contain negative values")
        batch_end = batch_start + 1
        while (
            batch_end < len(q_lens)
            and batch_end - batch_start < max_group_size
            and q_lens[batch_end] == q_count
        ):
            batch_end += 1
        if q_count > 0:
            groups.append((batch_start, batch_end, q_start, q_count))
        q_start += q_count * (batch_end - batch_start)
        batch_start = batch_end
    return tuple(groups)


def _scfa_q_count_bucket(q_count: int) -> int:
    """Round a positive runtime Q count to a reusable static loop bound."""
    if q_count <= 0:
        raise ValueError("SCFA Q count must be positive")
    return 1 << (q_count - 1).bit_length()


def _eager_batched_scfa_launch_plan(
    q_lens: list[int],
    kv_lens: list[int],
    *,
    cmp_storage: int,
    cmp_dtype: torch.dtype,
    worker_count: int,
    device: torch.device,
    stream_key: tuple[int, int],
) -> tuple[object, ...]:
    """Cache invariant slot and partition metadata for repeated DSV4 layers."""
    plan_key: tuple[object, ...] = (
        tuple(q_lens),
        tuple(kv_lens),
        cmp_storage,
        cmp_dtype,
        worker_count,
    )
    cached = _EAGER_BATCHED_SCFA_LAUNCH_PLAN_CACHE.get(stream_key)
    if cached is not None and cached[0] == plan_key:
        return cached[1]

    shared_slots = _allocate_scfa_pipeline_slots(
        device, cmp_dtype, worker_count, stream_key=stream_key
    )
    flat_slots = tuple(tensor for slot_group in shared_slots for tensor in slot_group)
    pipeline_options = _prefill_launch_options(multibuffer=True, unit_flag=False)
    pipeline_options["disable_auto_inject_block_sync"] = True
    batch_size = len(q_lens)
    grouped_q_lens = _group_consecutive_q_lens(q_lens, worker_count)
    group_plans = []
    if len(grouped_q_lens) > 1:
        # The non-partitioned kernel already uses q_plan to recover each token's
        # batch and causal position.  Let heterogeneous batches share one global
        # worker ring instead of paying one launch for every distinct q_len run.
        # A conservative N512 limit keeps long/sparse and short/dense sequences
        # valid in the same launch.
        total_q = sum(q_lens)
        pipeline_values = (
            batch_size * cmp_storage,
            cmp_storage,
            0,
            0,
            total_q,
            _scfa_q_count_bucket(total_q),
            512,
            worker_count,
            False,
            True,
            False,
            0,
            1,
            0,
            0,
            True,
        )
        group_plans.append(((worker_count,), pipeline_values))
    for batch_start, batch_end, q_start, q_count in (
        grouped_q_lens if len(grouped_q_lens) <= 1 else ()
    ):
        group_size = batch_end - batch_start
        group_kv_lens = kv_lens[batch_start:batch_end]
        workers_per_batch = worker_count // group_size
        partition_grid = (group_size * workers_per_batch,)
        dense_prefix = all(kv_len <= 2048 for kv_len in group_kv_lens)
        n_limit = (
            max(
                128,
                triton.cdiv(max(kv_len // 4 for kv_len in group_kv_lens), 128) * 128,
            )
            if dense_prefix
            else 512
        )
        pipeline_values = (
            batch_size * cmp_storage,
            cmp_storage,
            q_start,
            0,
            q_count * group_size,
            1,
            n_limit,
            worker_count,
            dense_prefix,
            True,
            True,
            q_count,
            _scfa_q_count_bucket(q_count),
            workers_per_batch,
            batch_start,
            True,
        )
        group_plans.append((partition_grid, pipeline_values))
    plan = (shared_slots, flat_slots, pipeline_options, tuple(group_plans))
    _EAGER_BATCHED_SCFA_LAUNCH_PLAN_CACHE[stream_key] = (plan_key, plan)
    return plan


def _use_compact_initial_prefix(q_lens: list[int], kv_lens: list[int]) -> bool:
    """Whether every sequence is an initial prompt fitting one N128 window."""
    return (
        len(q_lens) == len(kv_lens)
        and bool(q_lens)
        and all(0 < q_len == kv_len <= 128 for q_len, kv_len in zip(q_lens, kv_lens))
    )


def _use_exact_initial_page(q_lens: list[int], kv_lens: list[int]) -> bool:
    """Whether every sequence is an initial prompt filling one complete page."""
    return _use_compact_initial_prefix(q_lens, kv_lens) and all(
        q_len == 128 for q_len in q_lens
    )


def _launch_eager_batched_prefill(
    q: torch.Tensor,
    ori_kv: torch.Tensor,
    cmp_kv: Optional[torch.Tensor],
    cmp_sparse_indices: Optional[torch.Tensor],
    ori_block_table: torch.Tensor,
    cmp_block_table: Optional[torch.Tensor],
    cu_seqlens_q: torch.Tensor,
    seqused_kv: torch.Tensor,
    sinks: torch.Tensor,
    out: torch.Tensor,
    *,
    mode: int,
    q_lens: list[int],
    kv_lens: list[int],
    cmp_ratio: int,
    softmax_scale: float,
    stream_key: tuple[int, int],
) -> tuple[torch.Tensor, ...]:
    """一次launch链覆盖全部prefill序列，避免按batch重复启动。"""
    total_q = sum(q_lens)
    batch_size = len(q_lens)
    q_plan = _eager_q_plan(
        q_lens,
        kv_lens,
        cu_seqlens_q=cu_seqlens_q,
        seqused_kv=seqused_kv,
        device=q.device,
        stream_key=stream_key,
    )
    compact_initial_prefix = _use_compact_initial_prefix(q_lens, kv_lens)
    exact_initial_page = _use_exact_initial_page(q_lens, kv_lens)
    use_pack_plan = not compact_initial_prefix and batch_size >= 8
    prefix_pad = 0 if compact_initial_prefix else 127
    # An N128 read from the final sequence needs only one shared zero-filled tail.
    storage_count = total_q + (
        0
        if exact_initial_page
        else 127 if compact_initial_prefix else batch_size * prefix_pad
    )
    logical_ori_kv = _eager_workspace(
        "batched.ori_k",
        (storage_count, 512),
        dtype=ori_kv.dtype,
        device=q.device,
        stream_key=stream_key,
    )
    logical_ori_v = _eager_workspace(
        "batched.ori_v",
        (storage_count, 512),
        dtype=torch.float16,
        device=q.device,
        stream_key=stream_key,
    )
    grid = (get_num_cores("cube"),)
    ori_table_width = int(ori_block_table.shape[1])
    pack_plan = None
    if use_pack_plan:
        pack_plan = _eager_pack_plan(
            q_lens,
            kv_lens,
            cu_seqlens_q=cu_seqlens_q,
            seqused_kv=seqused_kv,
            prefix_pad=prefix_pad,
            storage_count=storage_count,
            device=q.device,
            stream_key=stream_key,
        )
    if exact_initial_page and mode == 0:
        pack_constexpr_names = (
            "TABLE_WIDTH",
            "HEAD_DIM",
            "PAGE_SIZE",
            "BLOCK_N",
            "BLOCK_D",
        )
        pack_constexpr_values = (ori_table_width, 512, 128, 128, 128)
        _launch_cached_eager_kernel(
            _sparse_attn_sharedkv_eager_batched_pack_exact_page_kernel,
            grid,
            (logical_ori_kv, logical_ori_v, ori_kv, ori_block_table, batch_size),
            pack_constexpr_names,
            pack_constexpr_values,
            _EAGER_BATCHED_PACK_OPTIONS,
            (ori_kv.dtype, ori_block_table.dtype, *pack_constexpr_values),
        )
    elif use_pack_plan:
        pack_constexpr_names = (
            "TABLE_WIDTH",
            "HEAD_DIM",
            "PAGE_SIZE",
            "BLOCK_N",
            "BLOCK_D",
        )
        pack_constexpr_values = (
            ori_table_width,
            512,
            128,
            64,
            256,
        )
        assert pack_plan is not None
        _launch_cached_eager_kernel(
            _sparse_attn_sharedkv_eager_batched_pack_original_planned_kernel,
            grid,
            (
                logical_ori_kv,
                logical_ori_v,
                ori_kv,
                ori_block_table,
                pack_plan,
                storage_count,
            ),
            pack_constexpr_names,
            pack_constexpr_values,
            _EAGER_BATCHED_PACK_OPTIONS,
            (
                ori_kv.dtype,
                ori_block_table.dtype,
                pack_plan.dtype,
                *pack_constexpr_values,
            ),
        )
    elif not exact_initial_page:
        pack_constexpr_names = (
            "TOTAL_Q",
            "STORAGE_COUNT",
            "BATCH_SIZE",
            "BATCH_BLOCK",
            "TABLE_WIDTH",
            "HEAD_DIM",
            "PAGE_SIZE",
            "PREFIX_PAD",
            "BLOCK_N",
            "BLOCK_D",
        )
        batch_block = max(2, triton.next_power_of_2(batch_size))
        pack_constexpr_values = (
            total_q,
            storage_count,
            batch_size,
            batch_block,
            ori_table_width,
            512,
            128,
            prefix_pad,
            64,
            256,
        )
        _launch_cached_eager_kernel(
            _sparse_attn_sharedkv_eager_batched_pack_original_kernel,
            grid,
            (
                logical_ori_kv,
                logical_ori_v,
                ori_kv,
                ori_block_table,
                cu_seqlens_q,
                seqused_kv,
            ),
            pack_constexpr_names,
            pack_constexpr_values,
            _EAGER_BATCHED_PACK_OPTIONS,
            (ori_kv.dtype, *pack_constexpr_values),
        )

    logical_cmp_kv = logical_ori_kv
    logical_cmp_v = logical_ori_v
    cmp_storage = 64
    workspaces: list[torch.Tensor] = [q_plan, logical_ori_kv, logical_ori_v]
    if pack_plan is not None:
        workspaces.append(pack_plan)
    if mode != 0:
        if cmp_kv is None or cmp_block_table is None:
            raise ValueError("compressed mode requires cmp_kv and cmp_block_table")
        cmp_storage = max(
            64,
            triton.cdiv(max((length // cmp_ratio for length in kv_lens), default=0), 64)
            * 64,
        )
        logical_cmp_kv = _eager_workspace(
            f"batched.cmp{cmp_ratio}.k",
            (batch_size * cmp_storage, 512),
            dtype=cmp_kv.dtype,
            device=q.device,
            stream_key=stream_key,
        )
        logical_cmp_v = _eager_workspace(
            f"batched.cmp{cmp_ratio}.v",
            (batch_size * cmp_storage, 512),
            dtype=torch.float16,
            device=q.device,
            stream_key=stream_key,
        )
        cmp_pack_constexpr_names = (
            "BATCH_SIZE",
            "STORAGE_STRIDE",
            "TABLE_WIDTH",
            "HEAD_DIM",
            "PAGE_SIZE",
            "CMP_RATIO",
            "BLOCK_N",
            "BLOCK_D",
        )
        cmp_pack_constexpr_values = (
            batch_size,
            cmp_storage,
            int(cmp_block_table.shape[1]),
            512,
            128,
            cmp_ratio,
            64,
            256,
        )
        if exact_initial_page:
            fused_pack_constexpr_names = (
                "ORI_TABLE_WIDTH",
                "CMP_TABLE_WIDTH",
                "HEAD_DIM",
                "PAGE_SIZE",
                "CMP_RATIO",
                "CMP_STORAGE",
                "BLOCK_N",
                "CMP_BLOCK_N",
                "BLOCK_D",
            )
            fused_pack_constexpr_values = (
                ori_table_width,
                int(cmp_block_table.shape[1]),
                512,
                128,
                cmp_ratio,
                cmp_storage,
                128,
                64,
                128,
            )
            _launch_q256_cached_eager_kernel(
                (
                    stream_key
                    if (
                        exact_initial_page
                        and 1 <= batch_size <= 8
                        and mode != 0
                        and q.dtype == ori_kv.dtype == cmp_kv.dtype == torch.bfloat16
                    )
                    else None
                ),
                _sparse_attn_sharedkv_eager_batched_pack_exact_page_with_compressed_kernel,
                grid,
                (
                    logical_ori_kv,
                    logical_ori_v,
                    logical_cmp_kv,
                    logical_cmp_v,
                    ori_kv,
                    cmp_kv,
                    ori_block_table,
                    cmp_block_table,
                    batch_size,
                ),
                fused_pack_constexpr_names,
                fused_pack_constexpr_values,
                _EAGER_BATCHED_PACK_OPTIONS,
                (
                    ori_kv.dtype,
                    cmp_kv.dtype,
                    ori_block_table.dtype,
                    cmp_block_table.dtype,
                    *fused_pack_constexpr_values,
                ),
            )
        else:
            _launch_cached_eager_kernel(
                _sparse_attn_sharedkv_eager_batched_pack_compressed_kernel,
                grid,
                (logical_cmp_kv, logical_cmp_v, cmp_kv, cmp_block_table, seqused_kv),
                cmp_pack_constexpr_names,
                cmp_pack_constexpr_values,
                _EAGER_BATCHED_PACK_OPTIONS,
                (cmp_kv.dtype, *cmp_pack_constexpr_values),
            )
        workspaces.extend((logical_cmp_kv, logical_cmp_v))

    # compressed候选超过一个N64时，SCFA不能把全部score与original score
    # 同时保留在M64 kernel中。PagePack和original partial仍合并整个batch；
    # compressed流水按连续等长q_len分组并按batch分区，既减少mixed batch的
    # launch数量，又保证一个program的event ring不跨越独立sequence时间轴。
    if mode == 2 and cmp_storage > 64:
        if cmp_sparse_indices is None:
            raise ValueError("SCFA batched prefill requires cmp_sparse_indices")
        ori_prob = _eager_workspace(
            "batched.scfa.ori_prob",
            (total_q * 64, 128),
            dtype=torch.float16,
            device=q.device,
            stream_key=stream_key,
        )
        ori_acc = _eager_workspace(
            "batched.scfa.ori_acc",
            (total_q, 64, 512),
            dtype=torch.float16,
            device=q.device,
            stream_key=stream_key,
        )
        ori_max = _eager_workspace(
            "batched.scfa.ori_max",
            (total_q, 64),
            dtype=torch.float32,
            device=q.device,
            stream_key=stream_key,
        )
        ori_sum = _eager_workspace(
            "batched.scfa.ori_sum",
            (total_q, 64),
            dtype=torch.float32,
            device=q.device,
            stream_key=stream_key,
        )
        stage_options = _EAGER_BATCHED_ATTENTION_OPTIONS
        ori_qk_constexpr_names = (
            "TOTAL_Q",
            "NUM_Q_HEADS",
            "HEAD_DIM",
            "PREFIX_PAD",
            "ORI_N",
            "D_BLOCK",
        )
        ori_qk_constexpr_values = (total_q, 64, 512, prefix_pad, 128, 256)
        _launch_cached_eager_kernel(
            _sparse_attn_sharedkv_eager_batched_scfa_original_qk_kernel,
            grid,
            (
                ori_prob,
                ori_max,
                ori_sum,
                q,
                logical_ori_kv,
                q_plan,
                float(softmax_scale),
            ),
            ori_qk_constexpr_names,
            ori_qk_constexpr_values,
            stage_options,
            (
                ori_prob.dtype,
                ori_max.dtype,
                q.dtype,
                logical_ori_kv.dtype,
                float(softmax_scale),
                *ori_qk_constexpr_values[1:],
            ),
        )
        ori_pv_constexpr_names = (
            "TOTAL_Q",
            "NUM_Q_HEADS",
            "HEAD_DIM",
            "PREFIX_PAD",
            "ORI_N",
            "CMP_STORAGE",
            "HAS_CMP",
        )
        ori_pv_constexpr_values = (total_q, 64, 512, prefix_pad, 128, 64, False)
        _launch_cached_eager_kernel(
            _sparse_attn_sharedkv_eager_batched_staged_pv_kernel,
            grid,
            (ori_acc, ori_prob, ori_prob, logical_ori_v, logical_ori_v, q_plan),
            ori_pv_constexpr_names,
            ori_pv_constexpr_values,
            stage_options,
            (
                ori_acc.dtype,
                ori_prob.dtype,
                logical_ori_v.dtype,
                *ori_pv_constexpr_values[1:],
            ),
        )

        worker_count = int(grid[0])
        shared_slots, flat_slots, pipeline_options, group_plans = (
            _eager_batched_scfa_launch_plan(
                q_lens,
                kv_lens,
                cmp_storage=cmp_storage,
                cmp_dtype=cmp_kv.dtype,
                worker_count=worker_count,
                device=q.device,
                stream_key=stream_key,
            )
        )
        workspaces.extend(flat_slots)
        (
            kv_slots,
            valid_slots,
            score_slots,
            prob_slots,
            acc_slots,
            max_slots,
            sum_slots,
        ) = shared_slots
        pipeline_args = (
            q,
            logical_cmp_kv,
            cmp_sparse_indices,
            q_plan,
            *kv_slots,
            *valid_slots,
            *score_slots,
            *prob_slots,
            *acc_slots,
            *max_slots,
            *sum_slots,
            ori_acc,
            ori_max,
            ori_sum,
            sinks,
            out,
            float(softmax_scale),
        )
        dispatch_signature = _batched_scfa_static_dispatch_signature(pipeline_args)
        for partition_grid, pipeline_values in group_plans:
            _launch_cached_scfa_score_guard_pipeline(
                partition_grid,
                pipeline_args,
                pipeline_values,
                pipeline_options,
                _scfa_static_slot_dispatch_key(dispatch_signature, pipeline_values),
            )
        workspaces.extend((ori_prob, ori_acc, ori_max, ori_sum))
        return tuple(workspaces)

    ori_prob = _eager_workspace(
        "batched.ori_prob",
        (total_q * 64, 128),
        dtype=torch.float16,
        device=q.device,
        stream_key=stream_key,
    )
    cmp_prob = ori_prob
    if mode != 0:
        cmp_prob = _eager_workspace(
            f"batched.cmp{cmp_ratio}.prob",
            (total_q * 64, cmp_storage),
            dtype=torch.float16,
            device=q.device,
            stream_key=stream_key,
        )
    workspaces.append(ori_prob)
    if mode != 0:
        workspaces.append(cmp_prob)

    attention_grid = grid
    qk_runtime_args = (
        ori_prob,
        cmp_prob,
        q,
        logical_ori_kv,
        logical_cmp_kv,
        q_plan,
        sinks,
        float(softmax_scale),
    )
    qk_constexpr_names = (
        "TOTAL_Q",
        "NUM_Q_HEADS",
        "HEAD_DIM",
        "PREFIX_PAD",
        "ORI_N",
        "CMP_STORAGE",
        "CMP_RATIO",
        "HAS_CMP",
        "D_BLOCK",
    )
    qk_constexpr_values = (
        total_q,
        64,
        512,
        prefix_pad,
        128,
        cmp_storage,
        max(cmp_ratio, 1),
        mode != 0,
        256,
    )
    _launch_q256_cached_eager_kernel(
        (
            stream_key
            if (
                exact_initial_page
                and 1 <= batch_size <= 8
                and mode != 0
                and q.dtype == ori_kv.dtype == cmp_kv.dtype == torch.bfloat16
            )
            else None
        ),
        _sparse_attn_sharedkv_eager_batched_staged_qk_softmax_kernel,
        attention_grid,
        qk_runtime_args,
        qk_constexpr_names,
        qk_constexpr_values,
        _EAGER_BATCHED_QK_OPTIONS,
        # TOTAL_Q is a runtime loop bound so MTP/async exact-Q variation can
        # reuse one compiled runner. All mathematical specialization remains.
        (q.dtype, ori_kv.dtype, qk_runtime_args[-1], *qk_constexpr_values[1:]),
    )
    pv_constexpr_names = (
        "TOTAL_Q",
        "NUM_Q_HEADS",
        "HEAD_DIM",
        "PREFIX_PAD",
        "ORI_N",
        "CMP_STORAGE",
        "HAS_CMP",
    )
    pv_constexpr_values = (total_q, 64, 512, prefix_pad, 128, cmp_storage, mode != 0)
    _launch_q256_cached_eager_kernel(
        (
            stream_key
            if (
                exact_initial_page
                and 1 <= batch_size <= 8
                and mode != 0
                and q.dtype == ori_kv.dtype == cmp_kv.dtype == torch.bfloat16
            )
            else None
        ),
        _sparse_attn_sharedkv_eager_batched_staged_pv_kernel,
        attention_grid,
        (out, ori_prob, cmp_prob, logical_ori_v, logical_cmp_v, q_plan),
        pv_constexpr_names,
        pv_constexpr_values,
        _EAGER_BATCHED_ATTENTION_OPTIONS,
        (out.dtype, ori_prob.dtype, logical_ori_v.dtype, *pv_constexpr_values[1:]),
    )
    return tuple(workspaces)


def _launch_chunked_swa_sequence(
    q: torch.Tensor,
    ori_kv: torch.Tensor,
    ori_block_table_row: torch.Tensor,
    sinks: torch.Tensor,
    out: torch.Tensor,
    *,
    kv_len: int,
    softmax_scale: float,
    stream_key: tuple[int, int],
) -> tuple[torch.Tensor, ...]:
    """单序列chunked SWA：一次Pack后让全部query复用M64/N128 staged计算。"""
    q_count = int(q.shape[0])
    plan_signature = None
    if (q_count == 1024 and kv_len == 8192) or (
        q_count == 256 and kv_len in range(256, 2049, 256)
    ):
        plan_cache = (
            _Q256_SWA_LAUNCH_PLAN_CACHE
            if q_count == 256
            else _Q1024_SWA_LAUNCH_PLAN_CACHE
        )
        plan_key = (*stream_key, kv_len) if q_count == 256 else stream_key
        plan_signature = (
            q.dtype,
            ori_kv.dtype,
            ori_block_table_row.dtype,
            int(ori_block_table_row.numel()),
            float(softmax_scale),
            sinks.dtype,
            out.dtype,
        )
        cached_plan = plan_cache.get(plan_key)
        if cached_plan is not None and cached_plan[0] == plan_signature:
            _, buffers, runners, constants = cached_plan
            logical_k, logical_v, prob = buffers
            pack_runner, qk_runner, pv_runner = runners
            pack_values, qk_values, pv_values = constants
            # Keep the original compiled runners, including runtime hooks. Only
            # workspaces and constants are retained; every input address is fresh.
            pack_runner(
                logical_k,
                logical_v,
                ori_kv,
                ori_block_table_row,
                *pack_values,
                stream=stream_key[1],
            )
            qk_runner(prob, q, logical_k, sinks, *qk_values, stream=stream_key[1])
            pv_runner(out, prob, logical_v, *pv_values, stream=stream_key[1])
            return buffers
    q_position_start = kv_len - q_count
    prefix_pad = 127
    storage_count = q_count + prefix_pad
    logical_k, logical_v, packed = _pack_chunked_kv_range(
        ori_kv,
        ori_block_table_row,
        logical_start=q_position_start - prefix_pad,
        token_count=storage_count,
        cache_tag="swa.ori",
        stream_key=stream_key,
        explicit_stream=plan_signature is not None,
    )
    prob = _eager_workspace(
        "swa.prob",
        (q_count * 64, 128),
        dtype=torch.float16,
        device=q.device,
        stream_key=stream_key,
    )
    grid = (get_num_cores("cube"),)
    qk_constexpr_names = (
        "TOTAL_Q",
        "NUM_Q_HEADS",
        "HEAD_DIM",
        "ORI_BLOCK_SIZE",
        "M_BLOCK",
        "N_BLOCK",
        "PROB_STRIDE",
        "D_BLOCK",
        "TOKEN_START",
        "TOKEN_END",
        "SOFTMAX_SCALE",
        "FULL_WINDOW",
        "TOKEN_BURST",
        "QK_LOOP_UNROLL",
        "USE_ROW_RECIPROCAL",
        "PREFIX_PAD",
        "Q_POSITION_OFFSET",
    )
    qk_constexpr_values = (
        q_count,
        64,
        512,
        128,
        64,
        128,
        128,
        256,
        0,
        q_count,
        float(softmax_scale),
        True,
        16,
        1,
        True,
        prefix_pad,
        q_position_start,
    )
    _launch_q256_cached_eager_kernel(
        stream_key if plan_signature is not None else None,
        _sparse_attn_sharedkv_prefill_swa_staged_qk_softmax_kernel,
        grid,
        (prob, q, logical_k, sinks),
        qk_constexpr_names,
        qk_constexpr_values,
        _EAGER_CHUNK_QK_OPTIONS,
        (q.dtype, logical_k.dtype, *qk_constexpr_values),
    )
    pv_constexpr_names = (
        "TOTAL_Q",
        "NUM_Q_HEADS",
        "HEAD_DIM",
        "ORI_BLOCK_SIZE",
        "M_BLOCK",
        "N_BLOCK",
        "PROB_STRIDE",
        "D_BLOCK",
        "TOKEN_START",
        "TOKEN_END",
        "FULL_WINDOW",
        "TOKEN_BURST",
        "PREFIX_PAD",
    )
    pv_constexpr_values = (
        q_count,
        64,
        512,
        128,
        64,
        128,
        128,
        512,
        0,
        q_count,
        True,
        64,
        prefix_pad,
    )
    _launch_q256_cached_eager_kernel(
        stream_key if plan_signature is not None else None,
        _sparse_attn_sharedkv_prefill_swa_staged_pv_kernel,
        grid,
        (out, prob, logical_v),
        pv_constexpr_names,
        pv_constexpr_values,
        _EAGER_BATCHED_PACK_OPTIONS,
        (q.dtype, *pv_constexpr_values),
    )
    if plan_signature is not None:
        pack_values = (
            q_position_start - prefix_pad,
            storage_count,
            storage_count,
            int(ori_block_table_row.numel()),
            512,
            128,
            64,
            256,
        )
        entries = (
            (
                _sparse_attn_sharedkv_chunk_pack_logical_k_and_fp16_v_kernel,
                (ori_kv.dtype, *pack_values),
            ),
            (
                _sparse_attn_sharedkv_prefill_swa_staged_qk_softmax_kernel,
                (q.dtype, logical_k.dtype, *qk_constexpr_values),
            ),
            (
                _sparse_attn_sharedkv_prefill_swa_staged_pv_kernel,
                (q.dtype, *pv_constexpr_values),
            ),
        )
        runners = tuple(
            _EAGER_COMPILED_LAUNCHER_CACHE.get((stream_key[0], id(kernel), grid, key))
            for kernel, key in entries
        )
        if all(callable(runner) for runner in runners):
            plan_cache[plan_key] = (
                plan_signature,
                (logical_k, logical_v, prob),
                runners,
                (pack_values, qk_constexpr_values, pv_constexpr_values),
            )
    return (*packed, prob)


def _launch_chunked_cfa_sequence(
    q: torch.Tensor,
    ori_kv: torch.Tensor,
    cmp_kv: torch.Tensor,
    ori_block_table_row: torch.Tensor,
    cmp_block_table_row: torch.Tensor,
    sinks: torch.Tensor,
    out: torch.Tensor,
    *,
    kv_len: int,
    softmax_scale: float,
    stream_key: tuple[int, int],
) -> tuple[torch.Tensor, ...]:
    """单序列chunked CFA：original窗口和至多N64 compressed前缀统一softmax。"""
    q_count = int(q.shape[0])
    plan_signature = None
    if (q_count == 1024 and kv_len == 8192) or (
        q_count == 256 and kv_len in range(256, 2049, 256)
    ):
        # Keep eight q256 chunk plans per stream so consecutive KV lengths do
        # not evict one another. The existing q1024 cache stays independent.
        plan_cache = (
            _Q256_CFA_LAUNCH_PLAN_CACHE
            if q_count == 256
            else _Q1024_CFA_LAUNCH_PLAN_CACHE
        )
        plan_key = (*stream_key, kv_len) if q_count == 256 else stream_key
        plan_signature = (
            q.dtype,
            ori_kv.dtype,
            cmp_kv.dtype,
            ori_block_table_row.dtype,
            cmp_block_table_row.dtype,
            int(ori_block_table_row.numel()),
            int(cmp_block_table_row.numel()),
            float(softmax_scale),
            sinks.dtype,
            out.dtype,
        )
        cached_plan = plan_cache.get(plan_key)
        if cached_plan is not None and cached_plan[0] == plan_signature:
            _, buffers, runners, values = cached_plan
            (
                logical_ori_k,
                logical_ori_v,
                logical_cmp_k,
                logical_cmp_v,
                ori_prob,
                cmp_prob,
            ) = buffers
            pack_runner, qk_runner, pv_runner = runners
            pack_values, qk_values, pv_values = values
            pack_runner(
                logical_ori_k,
                logical_ori_v,
                logical_cmp_k,
                logical_cmp_v,
                ori_kv,
                cmp_kv,
                ori_block_table_row,
                cmp_block_table_row,
                *pack_values,
                stream=stream_key[1],
            )
            qk_runner(
                ori_prob,
                cmp_prob,
                q,
                logical_ori_k,
                logical_cmp_k,
                sinks,
                *qk_values,
                stream=stream_key[1],
            )
            pv_runner(
                out,
                ori_prob,
                cmp_prob,
                logical_ori_v,
                logical_cmp_v,
                *pv_values,
                stream=stream_key[1],
            )
            return buffers
    q_position_start = kv_len - q_count
    prefix_pad = 127
    cmp_count = kv_len // 128
    if (q_count == 256 and kv_len in range(256, 2049, 256)) or (
        q_count == 1024 and kv_len == 8192
    ):
        (
            logical_ori_k,
            logical_ori_v,
            logical_cmp_k,
            logical_cmp_v,
            fused_packed,
        ) = _pack_q256_ori_cmp_ranges(
            ori_kv,
            cmp_kv,
            ori_block_table_row,
            cmp_block_table_row,
            ori_logical_start=q_position_start - prefix_pad,
            ori_token_count=q_count + prefix_pad,
            ori_storage_count=q_count + prefix_pad,
            cmp_token_count=cmp_count,
            cmp_storage_count=128,
            write_cmp_v=True,
            cache_tag="cfa.q256_fused",
            stream_key=stream_key,
        )
        ori_packed = fused_packed
        cmp_packed = ()
    else:
        logical_ori_k, logical_ori_v, ori_packed = _pack_chunked_kv_range(
            ori_kv,
            ori_block_table_row,
            logical_start=q_position_start - prefix_pad,
            token_count=q_count + prefix_pad,
            cache_tag="cfa.ori",
            stream_key=stream_key,
        )
        # token_count=0同样启动pack，由kernel把完整N128 storage写0；这样不需要
        # 单独的torch.zeros分支，后续QK/PV始终读取稳定地址。
        logical_cmp_k, logical_cmp_v, cmp_packed = _pack_chunked_kv_range(
            cmp_kv,
            cmp_block_table_row,
            logical_start=0,
            token_count=cmp_count,
            storage_count=128,
            cache_tag="cfa.cmp",
            stream_key=stream_key,
        )

    ori_prob = _eager_workspace(
        "cfa.ori_prob",
        (q_count * 64, 128),
        dtype=torch.float16,
        device=q.device,
        stream_key=stream_key,
    )
    cmp_prob = _eager_workspace(
        "cfa.cmp_prob",
        (q_count * 64, 64),
        dtype=torch.float16,
        device=q.device,
        stream_key=stream_key,
    )
    grid = (get_num_cores("cube"),)
    qk_constexpr_names = (
        "TOTAL_Q",
        "NUM_Q_HEADS",
        "HEAD_DIM",
        "ORI_BLOCK_SIZE",
        "CMP_BLOCK_SIZE",
        "CMP_RATIO",
        "M_BLOCK",
        "ORI_N_BLOCK",
        "CMP_N_BLOCK",
        "D_BLOCK",
        "TOKEN_START",
        "TOKEN_END",
        "SOFTMAX_SCALE",
        "FULL_ORI_WINDOW",
        "TOKEN_BURST",
        "QK_LOOP_UNROLL",
        "USE_ROW_RECIPROCAL",
        "Q_POSITION_OFFSET",
        "PREFIX_PAD",
    )
    qk_constexpr_values = (
        q_count,
        64,
        512,
        128,
        128,
        128,
        64,
        128,
        64,
        256,
        0,
        q_count,
        float(softmax_scale),
        True,
        16,
        1,
        True,
        q_position_start,
        prefix_pad,
    )
    _launch_q256_cached_eager_kernel(
        (
            stream_key
            if (
                (q_count == 256 and kv_len in range(256, 2049, 256))
                or (q_count == 1024 and kv_len == 8192)
            )
            else None
        ),
        _sparse_attn_sharedkv_prefill_cfa_staged_qk_softmax_kernel,
        grid,
        (ori_prob, cmp_prob, q, logical_ori_k, logical_cmp_k, sinks),
        qk_constexpr_names,
        qk_constexpr_values,
        _EAGER_BATCHED_PACK_OPTIONS,
        (q.dtype, logical_ori_k.dtype, logical_cmp_k.dtype, *qk_constexpr_values),
    )
    pv_constexpr_names = (
        "TOTAL_Q",
        "NUM_Q_HEADS",
        "HEAD_DIM",
        "ORI_BLOCK_SIZE",
        "CMP_BLOCK_SIZE",
        "M_BLOCK",
        "ORI_N_BLOCK",
        "CMP_N_BLOCK",
        "D_BLOCK",
        "TOKEN_START",
        "TOKEN_END",
        "FULL_ORI_WINDOW",
        "TOKEN_BURST",
        "PREFIX_PAD",
    )
    pv_constexpr_values = (
        q_count,
        64,
        512,
        128,
        128,
        64,
        128,
        64,
        512,
        0,
        q_count,
        True,
        16,
        prefix_pad,
    )
    _launch_q256_cached_eager_kernel(
        (
            stream_key
            if (
                (q_count == 256 and kv_len in range(256, 2049, 256))
                or (q_count == 1024 and kv_len == 8192)
            )
            else None
        ),
        _sparse_attn_sharedkv_prefill_cfa_staged_pv_kernel,
        grid,
        (out, ori_prob, cmp_prob, logical_ori_v, logical_cmp_v),
        pv_constexpr_names,
        pv_constexpr_values,
        _EAGER_CHUNK_PV_UNIT0_OPTIONS,
        (q.dtype, *pv_constexpr_values),
    )
    if plan_signature is not None:
        pack_values = (
            q_position_start - prefix_pad,
            q_count + prefix_pad,
            q_count + prefix_pad,
            cmp_count,
            128,
            int(ori_block_table_row.numel()),
            int(cmp_block_table_row.numel()),
            512,
            128,
            64,
            256,
            True,
        )
        entries = (
            (
                _sparse_attn_sharedkv_q256_fused_ori_cmp_pack_kernel,
                (
                    ori_kv.dtype,
                    cmp_kv.dtype,
                    ori_block_table_row.dtype,
                    cmp_block_table_row.dtype,
                    *pack_values,
                ),
            ),
            (
                _sparse_attn_sharedkv_prefill_cfa_staged_qk_softmax_kernel,
                (
                    q.dtype,
                    logical_ori_k.dtype,
                    logical_cmp_k.dtype,
                    *qk_constexpr_values,
                ),
            ),
            (
                _sparse_attn_sharedkv_prefill_cfa_staged_pv_kernel,
                (q.dtype, *pv_constexpr_values),
            ),
        )
        runners = tuple(
            _EAGER_COMPILED_LAUNCHER_CACHE.get((stream_key[0], id(kernel), grid, key))
            for kernel, key in entries
        )
        if all(callable(runner) for runner in runners):
            plan_cache[plan_key] = (
                plan_signature,
                (
                    logical_ori_k,
                    logical_ori_v,
                    logical_cmp_k,
                    logical_cmp_v,
                    ori_prob,
                    cmp_prob,
                ),
                runners,
                (pack_values, qk_constexpr_values, pv_constexpr_values),
            )
    return (*ori_packed, *cmp_packed, ori_prob, cmp_prob)


def _allocate_scfa_pipeline_slots(
    device: torch.device,
    dtype: torch.dtype,
    worker_count: int,
    *,
    stream_key: Optional[tuple[int, int]] = None,
) -> tuple[list[torch.Tensor], ...]:
    """分配可在同一stream上由多个batch顺序复用的SCFA静态流水槽。"""
    resolved_stream_key = stream_key or _eager_stream_key(device)
    kv_slots = [
        _eager_workspace(
            f"scfa.slot.kv{index}",
            (worker_count, 512, 512),
            dtype=dtype,
            device=device,
            stream_key=resolved_stream_key,
        )
        for index in range(4)
    ]
    valid_slots = [
        _eager_workspace(
            f"scfa.slot.valid{index}",
            (worker_count, 512),
            dtype=torch.uint8,
            device=device,
            stream_key=resolved_stream_key,
        )
        for index in range(4)
    ]
    score_slots = [
        _eager_workspace(
            f"scfa.slot.score{index}",
            (worker_count, 64, 512),
            dtype=torch.float32,
            device=device,
            stream_key=resolved_stream_key,
        )
        for index in range(2)
    ]
    prob_slots = [
        _eager_workspace(
            f"scfa.slot.prob{index}",
            (worker_count, 64, 512),
            dtype=dtype,
            device=device,
            stream_key=resolved_stream_key,
        )
        for index in range(4)
    ]
    acc_slots = [
        _eager_workspace(
            f"scfa.slot.acc{index}",
            (worker_count, 64, 512),
            dtype=torch.float16,
            device=device,
            stream_key=resolved_stream_key,
        )
        for index in range(2)
    ]
    max_slots = [
        _eager_workspace(
            f"scfa.slot.max{index}",
            (worker_count, 64),
            dtype=torch.float32,
            device=device,
            stream_key=resolved_stream_key,
        )
        for index in range(4)
    ]
    sum_slots = [
        _eager_workspace(
            f"scfa.slot.sum{index}",
            (worker_count, 64),
            dtype=torch.float32,
            device=device,
            stream_key=resolved_stream_key,
        )
        for index in range(4)
    ]
    return (
        kv_slots,
        valid_slots,
        score_slots,
        prob_slots,
        acc_slots,
        max_slots,
        sum_slots,
    )


def _launch_chunked_scfa_sequence(
    q: torch.Tensor,
    ori_kv: torch.Tensor,
    cmp_kv: torch.Tensor,
    cmp_sparse_indices: torch.Tensor,
    ori_block_table_row: torch.Tensor,
    cmp_block_table_row: torch.Tensor,
    sinks: torch.Tensor,
    out: torch.Tensor,
    shared_slots: tuple[list[torch.Tensor], ...],
    *,
    kv_len: int,
    softmax_scale: float,
    stream_key: tuple[int, int],
) -> tuple[torch.Tensor, ...]:
    """单序列chunked SCFA：复用已验证形状的 original staged 与 compressed CV 流水。"""
    q_count = int(q.shape[0])
    plan_signature = None
    if q_count == 256 and kv_len in range(256, 2049, 256):
        divisibility = (
            _sparse_attn_sharedkv_prefill_scfa_static_slot_pipeline_kernel.divisibility
        )
        plan_key = (*stream_key, kv_len)
        plan_signature = (
            q.dtype,
            ori_kv.dtype,
            cmp_kv.dtype,
            cmp_sparse_indices.dtype,
            ori_block_table_row.dtype,
            cmp_block_table_row.dtype,
            int(ori_block_table_row.numel()),
            int(cmp_block_table_row.numel()),
            sinks.dtype,
            out.dtype,
            float(softmax_scale),
            q.data_ptr() % divisibility == 0,
            cmp_sparse_indices.data_ptr() % divisibility == 0,
            sinks.data_ptr() % divisibility == 0,
            id(shared_slots),
        )
        cached_plan = _Q256_SCFA_LAUNCH_PLAN_CACHE.get(plan_key)
        if cached_plan is not None and cached_plan[0] == plan_signature:
            _, buffers, runners, values, pipeline_middle, retained_slots = cached_plan
            (
                logical_ori_k,
                logical_ori_v,
                logical_cmp_kv,
                ori_prob,
                ori_acc,
                ori_max,
                ori_sum,
            ) = buffers
            pack_runner, qk_runner, pv_runner, pipeline_runner = runners
            pack_values, qk_values, pv_values, pipeline_values = values
            pack_runner(
                logical_ori_k,
                logical_ori_v,
                logical_cmp_kv,
                logical_cmp_kv,
                ori_kv,
                cmp_kv,
                ori_block_table_row,
                cmp_block_table_row,
                *pack_values,
                stream=stream_key[1],
            )
            qk_runner(
                ori_prob,
                ori_max,
                ori_sum,
                q,
                logical_ori_k,
                *qk_values,
                stream=stream_key[1],
            )
            pv_runner(
                ori_acc, ori_prob, logical_ori_v, *pv_values, stream=stream_key[1]
            )
            pipeline_runner(
                q,
                logical_cmp_kv,
                cmp_sparse_indices,
                q,
                *pipeline_middle,
                sinks,
                out,
                float(softmax_scale),
                *pipeline_values,
                stream=stream_key[1],
            )
            return buffers
    q_position_start = kv_len - q_count
    prefix_pad = 127
    worker_count = get_num_cores("cube")
    grid = (worker_count,)
    launch_options = _EAGER_SCFA_PIPELINE_OPTIONS
    stage_options = _EAGER_SCFA_STAGE_OPTIONS

    cmp_count = kv_len // 4
    cmp_storage_count = max(128, triton.cdiv(max(cmp_count, 1), 128) * 128)
    fused_q256_pack = q_count == 256 and kv_len in range(256, 2049, 256)
    if fused_q256_pack:
        (
            logical_ori_k,
            logical_ori_v,
            logical_cmp_kv,
            _,
            ori_packed,
        ) = _pack_q256_ori_cmp_ranges(
            ori_kv,
            cmp_kv,
            ori_block_table_row,
            cmp_block_table_row,
            ori_logical_start=q_position_start - prefix_pad,
            ori_token_count=q_count + prefix_pad,
            ori_storage_count=q_count + prefix_pad,
            cmp_token_count=cmp_count,
            cmp_storage_count=cmp_storage_count,
            write_cmp_v=False,
            cache_tag="scfa.q256_fused",
            stream_key=stream_key,
        )
    else:
        logical_ori_k, logical_ori_v, ori_packed = _pack_chunked_kv_range(
            ori_kv,
            ori_block_table_row,
            logical_start=q_position_start - prefix_pad,
            token_count=q_count + prefix_pad,
            cache_tag="scfa.ori",
            stream_key=stream_key,
        )
    ori_prob = _eager_workspace(
        "scfa.ori_prob",
        (q_count * 64, 128),
        dtype=torch.float16,
        device=q.device,
        stream_key=stream_key,
    )
    ori_acc = _eager_workspace(
        "scfa.ori_acc",
        (q_count, 64, 512),
        dtype=torch.float16,
        device=q.device,
        stream_key=stream_key,
    )
    ori_max = _eager_workspace(
        "scfa.ori_max",
        (q_count, 64),
        dtype=torch.float32,
        device=q.device,
        stream_key=stream_key,
    )
    ori_sum = _eager_workspace(
        "scfa.ori_sum",
        (q_count, 64),
        dtype=torch.float32,
        device=q.device,
        stream_key=stream_key,
    )
    ori_qk_constexpr_names = (
        "TOTAL_Q",
        "NUM_Q_HEADS",
        "HEAD_DIM",
        "ORI_BLOCK_SIZE",
        "M_BLOCK",
        "N_BLOCK",
        "D_BLOCK",
        "TOKEN_START",
        "TOKEN_END",
        "SOFTMAX_SCALE",
        "FULL_WINDOW",
        "TOKEN_BURST",
        "Q_POSITION_OFFSET",
        "PREFIX_PAD",
    )
    ori_qk_constexpr_values = (
        q_count,
        64,
        512,
        128,
        64,
        128,
        256,
        0,
        q_count,
        float(softmax_scale),
        True,
        16,
        q_position_start,
        prefix_pad,
    )
    _launch_q256_cached_eager_kernel(
        stream_key if fused_q256_pack else None,
        _sparse_attn_sharedkv_prefill_scfa_window_qk_partial_kernel,
        grid,
        (ori_prob, ori_max, ori_sum, q, logical_ori_k),
        ori_qk_constexpr_names,
        ori_qk_constexpr_values,
        stage_options,
        (q.dtype, logical_ori_k.dtype, *ori_qk_constexpr_values),
    )
    ori_pv_constexpr_names = (
        "TOTAL_Q",
        "NUM_Q_HEADS",
        "HEAD_DIM",
        "ORI_BLOCK_SIZE",
        "M_BLOCK",
        "N_BLOCK",
        "PROB_STRIDE",
        "D_BLOCK",
        "TOKEN_START",
        "TOKEN_END",
        "FULL_WINDOW",
        "TOKEN_BURST",
        "PREFIX_PAD",
    )
    ori_pv_constexpr_values = (
        q_count,
        64,
        512,
        128,
        64,
        128,
        128,
        512,
        0,
        q_count,
        True,
        64,
        prefix_pad,
    )
    _launch_q256_cached_eager_kernel(
        stream_key if fused_q256_pack else None,
        _sparse_attn_sharedkv_prefill_swa_staged_pv_kernel,
        grid,
        (ori_acc, ori_prob, logical_ori_v),
        ori_pv_constexpr_names,
        ori_pv_constexpr_values,
        stage_options,
        (
            ori_acc.dtype,
            ori_prob.dtype,
            logical_ori_v.dtype,
            *ori_pv_constexpr_values,
        ),
    )

    if not fused_q256_pack:
        logical_cmp_kv = _eager_workspace(
            "scfa.cmp_kv",
            (cmp_storage_count, 512),
            dtype=cmp_kv.dtype,
            device=q.device,
            stream_key=stream_key,
        )
        cmp_pack_constexpr_names = ("TOTAL_Q", "HEAD_DIM", "ORI_BLOCK_SIZE", "D_BLOCK")
        cmp_pack_constexpr_values = (cmp_storage_count, 512, 128, 256)
        _launch_cached_eager_kernel(
            _sparse_attn_sharedkv_prefill_swa_pack_logical_kv_kernel,
            grid,
            (logical_cmp_kv, cmp_kv, cmp_block_table_row),
            cmp_pack_constexpr_names,
            cmp_pack_constexpr_values,
            launch_options,
            (cmp_kv.dtype, *cmp_pack_constexpr_values),
        )

    dense_prefix = kv_len <= 2048
    n_limit = (
        max(128, triton.cdiv(max(cmp_count, 1), 128) * 128) if dense_prefix else 512
    )
    kv_slots, valid_slots, score_slots, prob_slots, acc_slots, max_slots, sum_slots = (
        shared_slots
    )
    pipeline_args = (
        q,
        logical_cmp_kv,
        cmp_sparse_indices,
        q,
        *kv_slots,
        *valid_slots,
        *score_slots,
        *prob_slots,
        *acc_slots,
        *max_slots,
        *sum_slots,
        ori_acc,
        ori_max,
        ori_sum,
        sinks,
        out,
        float(softmax_scale),
    )
    pipeline_values = (
        cmp_storage_count,
        cmp_storage_count,
        0,
        q_position_start,
        q_count,
        _scfa_q_count_bucket(q_count),
        n_limit,
        worker_count,
        dense_prefix,
        False,
        False,
        0,
        1,
        0,
        0,
        True,
    )
    pipeline_options = dict(launch_options)
    pipeline_options["disable_auto_inject_block_sync"] = True
    _launch_q256_cached_scfa_score_guard_pipeline(
        stream_key if fused_q256_pack else None,
        grid,
        pipeline_args,
        pipeline_values,
        pipeline_options,
        _scfa_static_slot_dispatch_key(
            _single_scfa_static_dispatch_signature(pipeline_args),
            pipeline_values,
        ),
    )
    buffers = (*ori_packed, logical_cmp_kv, ori_prob, ori_acc, ori_max, ori_sum)
    if plan_signature is not None:
        pack_values = (
            q_position_start - prefix_pad,
            q_count + prefix_pad,
            q_count + prefix_pad,
            cmp_count,
            cmp_storage_count,
            int(ori_block_table_row.numel()),
            int(cmp_block_table_row.numel()),
            512,
            128,
            64,
            256,
            False,
        )
        keys = (
            (
                _sparse_attn_sharedkv_q256_fused_ori_cmp_pack_kernel,
                (
                    ori_kv.dtype,
                    cmp_kv.dtype,
                    ori_block_table_row.dtype,
                    cmp_block_table_row.dtype,
                    *pack_values,
                ),
            ),
            (
                _sparse_attn_sharedkv_prefill_scfa_window_qk_partial_kernel,
                (q.dtype, logical_ori_k.dtype, *ori_qk_constexpr_values),
            ),
            (
                _sparse_attn_sharedkv_prefill_swa_staged_pv_kernel,
                (
                    ori_acc.dtype,
                    ori_prob.dtype,
                    logical_ori_v.dtype,
                    *ori_pv_constexpr_values,
                ),
            ),
            (
                _sparse_attn_sharedkv_prefill_scfa_static_slot_pipeline_kernel,
                _scfa_static_slot_dispatch_key(
                    _single_scfa_static_dispatch_signature(pipeline_args),
                    pipeline_values,
                ),
            ),
        )
        runners = tuple(
            _EAGER_COMPILED_LAUNCHER_CACHE.get((stream_key[0], id(kernel), grid, key))
            for kernel, key in keys
        )
        if all(callable(runner) for runner in runners):
            # Retain slots to prevent identity reuse; no caller tensors are cached.
            plan_buffers = (
                logical_ori_k,
                logical_ori_v,
                logical_cmp_kv,
                ori_prob,
                ori_acc,
                ori_max,
                ori_sum,
            )
            _Q256_SCFA_LAUNCH_PLAN_CACHE[plan_key] = (
                plan_signature,
                plan_buffers,
                runners,
                (
                    pack_values,
                    ori_qk_constexpr_values,
                    ori_pv_constexpr_values,
                    pipeline_values,
                ),
                pipeline_args[4:31],
                shared_slots,
            )
    return buffers


def _launch_q256_cached_scfa_score_guard_pipeline(
    stream_key: Optional[tuple[int, int]],
    grid: tuple[int, ...],
    positional_args: tuple[object, ...],
    trailing_values: tuple[object, ...],
    compile_options: dict[str, object],
    dispatch_key: tuple[object, ...],
) -> None:
    """Keep interleaved runtime/constexpr ordering on the explicit-stream path."""
    if len(trailing_values) != len(_SCFA_STATIC_SLOT_TRAILING_NAMES):
        raise ValueError("SCFA static-slot launcher argument count mismatch")
    if stream_key is not None:
        kernel_entry = _sparse_attn_sharedkv_prefill_scfa_static_slot_pipeline_kernel
        cache_key = (stream_key[0], id(kernel_entry), grid, dispatch_key)
        cached = _EAGER_COMPILED_LAUNCHER_CACHE.get(cache_key)
        if cached is not None and cached is not _EAGER_COMPILED_LAUNCHER_UNAVAILABLE:
            cached(*(positional_args + trailing_values), stream=stream_key[1])
            return
    _launch_cached_scfa_score_guard_pipeline(
        grid,
        positional_args,
        trailing_values,
        compile_options,
        dispatch_key,
    )


def _launch_cached_scfa_score_guard_pipeline(
    grid: tuple[int, ...],
    positional_args: tuple[object, ...],
    trailing_values: tuple[object, ...],
    compile_options: dict[str, object],
    dispatch_key: tuple[object, ...],
) -> None:
    """Cache the event-pipeline runner without changing its public signature.

    This kernel interleaves runtime scalar and constexpr arguments after its tensor
    pointers.  Keep the first LibEntry call keyword-based, then invoke the cached
    runner with values restored to the kernel's original positional order.
    """
    kernel_entry = _sparse_attn_sharedkv_prefill_scfa_static_slot_pipeline_kernel
    if len(trailing_values) != len(_SCFA_STATIC_SLOT_TRAILING_NAMES):
        raise ValueError("SCFA static-slot launcher argument count mismatch")
    device_index = torch.npu.current_device()
    cache_key = (device_index, id(kernel_entry), grid, dispatch_key)
    cached = _EAGER_COMPILED_LAUNCHER_CACHE.get(cache_key)
    ordered_args = positional_args + trailing_values
    if cached is not None and cached is not _EAGER_COMPILED_LAUNCHER_UNAVAILABLE:
        cached(*ordered_args)
        return

    launch_kwargs = dict(zip(_SCFA_STATIC_SLOT_TRAILING_NAMES, trailing_values))
    launch_kwargs.update(compile_options)
    kernel_entry[grid](*positional_args, **launch_kwargs)
    if cached is _EAGER_COMPILED_LAUNCHER_UNAVAILABLE:
        return

    try:
        specialize_indices = set(kernel_entry.specialize_indices)
        do_not_specialize_indices = set(kernel_entry.do_not_specialize_indices)
        spec_args = [
            arg for index, arg in enumerate(ordered_args) if index in specialize_indices
        ]
        dns_args = [
            arg
            for index, arg in enumerate(ordered_args)
            if index in do_not_specialize_indices
        ]
        const_args = [
            arg
            for index, arg in enumerate(ordered_args)
            if index not in specialize_indices
            and index not in do_not_specialize_indices
        ]
        entry_key = kernel_entry.key(spec_args, dns_args, const_args)
        compiled_kernel = kernel_entry.kernel_cache[device_index][entry_key][0]
        grid3 = (grid + (1, 1))[:3]
        _EAGER_COMPILED_LAUNCHER_CACHE[cache_key] = compiled_kernel[grid3]
    except (AttributeError, KeyError, TypeError):
        _EAGER_COMPILED_LAUNCHER_CACHE[cache_key] = _EAGER_COMPILED_LAUNCHER_UNAVAILABLE


def sparse_attn_sharedkv_eager_prefill_impl(
    q: torch.Tensor,
    *,
    ori_kv: torch.Tensor,
    cmp_kv: Optional[torch.Tensor] = None,
    ori_sparse_indices: Optional[torch.Tensor] = None,
    cmp_sparse_indices: Optional[torch.Tensor] = None,
    ori_block_table: torch.Tensor,
    cmp_block_table: Optional[torch.Tensor] = None,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_ori_kv: Optional[torch.Tensor] = None,
    cu_seqlens_cmp_kv: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_kv: torch.Tensor,
    sinks: torch.Tensor,
    metadata: Optional[torch.Tensor] = None,
    softmax_scale: float = 0,
    cmp_ratio: int = 0,
    ori_mask_mode: int = 4,
    cmp_mask_mode: int = 3,
    ori_win_left: int = 127,
    ori_win_right: int = 0,
    layout_q: str = "TND",
    layout_kv: str = "PA_ND",
    return_softmax_lse: bool = False,
    max_model_len: Optional[int] = None,
    host_q_lens: Optional[list[int]] = None,
    host_kv_lens: Optional[list[int]] = None,
) -> torch.Tensor:
    """Chunked prefill using the caller's host lengths, without device readback."""
    global _LAST_EAGER_WORKSPACE
    del cu_seqlens_ori_kv, cu_seqlens_cmp_kv, seqused_q, metadata
    if return_softmax_lse:
        raise NotImplementedError("softmax_lse is not supported.")
    if layout_q != "TND" or layout_kv != "PA_ND":
        raise NotImplementedError("The eager prefill path requires TND Q and PA_ND KV.")
    if q.dim() != 3 or tuple(q.shape[1:]) != (64, 512):
        raise ValueError(f"q must have shape [T,64,512], got {tuple(q.shape)}")
    if ori_sparse_indices is not None and ori_sparse_indices.numel() > 0:
        raise ValueError("ori_sparse_indices is not supported.")
    if (
        ori_mask_mode != 4
        or cmp_mask_mode != 3
        or ori_win_left != 127
        or ori_win_right != 0
    ):
        raise ValueError(
            "Only DSV4 mask modes 4/3 and SWA window [127,0] are supported."
        )
    if softmax_scale in (None, 0):
        softmax_scale = 512**-0.5

    if (host_q_lens is None) != (host_kv_lens is None):
        raise ValueError("host_q_lens and host_kv_lens must be provided together.")
    if host_q_lens is None:
        raise ValueError("Eager prefill requires host_q_lens and host_kv_lens.")
    else:
        q_lens = [int(value) for value in host_q_lens]
        kv_lens = [int(value) for value in host_kv_lens]
        cu_q = [0]
        for q_len in q_lens:
            cu_q.append(cu_q[-1] + q_len)
    batch_size = len(cu_q) - 1
    if batch_size <= 0 or len(kv_lens) != batch_size:
        raise ValueError("cu_seqlens_q and seqused_kv batch dimensions must agree.")
    active_q = cu_q[-1]
    if active_q < 0 or active_q > int(q.shape[0]):
        raise ValueError("cu_seqlens_q[-1] is outside q capacity.")
    if any(length < 0 for length in q_lens):
        raise ValueError("cu_seqlens_q must be monotonic.")
    if any(kv_len < q_len for kv_len, q_len in zip(kv_lens, q_lens)):
        raise ValueError("seqused_kv cannot be shorter than the current query segment.")

    has_cmp = isinstance(cmp_kv, torch.Tensor) and cmp_kv.numel() > 0
    has_sparse = (
        isinstance(cmp_sparse_indices, torch.Tensor) and cmp_sparse_indices.numel() > 0
    )
    mode = 0 if not has_cmp else (2 if has_sparse else 1)
    if mode == 1 and any(kv_len // 128 > 64 for kv_len in kv_lens):
        return sparse_attn_sharedkv_graphsafe_impl(
            q,
            ori_kv=ori_kv,
            cmp_kv=cmp_kv,
            ori_sparse_indices=ori_sparse_indices,
            cmp_sparse_indices=cmp_sparse_indices,
            ori_block_table=ori_block_table,
            cmp_block_table=cmp_block_table,
            cu_seqlens_q=cu_seqlens_q,
            seqused_kv=seqused_kv,
            sinks=sinks,
            softmax_scale=softmax_scale,
            cmp_ratio=cmp_ratio,
            ori_mask_mode=ori_mask_mode,
            cmp_mask_mode=cmp_mask_mode,
            ori_win_left=ori_win_left,
            ori_win_right=ori_win_right,
            layout_q=layout_q,
            layout_kv=layout_kv,
            max_model_len=max_model_len,
        )

    if (
        mode == 2
        and batch_size == 1
        and q_lens[0] == 256
        and int(q.shape[0]) == 256
        and kv_lens[0] in range(256, 2049, 256)
        and ori_block_table.dim() == 2
        and int(ori_block_table.shape[0]) == 1
        and cmp_block_table.dim() == 2
        and int(cmp_block_table.shape[0]) == 1
    ):
        stream_key = _eager_raw_stream_key(q.device)
        out = _eager_workspace(
            "single.out",
            tuple(q.shape),
            dtype=q.dtype,
            device=q.device,
            stream_key=stream_key,
        )
        worker_count = get_num_cores("cube")
        slot_key = (*stream_key, cmp_kv.dtype, worker_count)
        slots = _MODEL_SINGLE_SCFA_SLOT_CACHE.get(slot_key)
        if slots is None:
            slots = _allocate_scfa_pipeline_slots(
                q.device, cmp_kv.dtype, worker_count, stream_key=stream_key
            )
            _MODEL_SINGLE_SCFA_SLOT_CACHE[slot_key] = slots
        buffers = _launch_chunked_scfa_sequence(
            q,
            ori_kv,
            cmp_kv,
            (
                cmp_sparse_indices
                if int(cmp_sparse_indices.shape[0]) == 256
                else cmp_sparse_indices[:256]
            ),
            ori_block_table,
            cmp_block_table,
            sinks,
            out,
            slots,
            kv_len=kv_lens[0],
            softmax_scale=float(softmax_scale),
            stream_key=stream_key,
        )
        _LAST_EAGER_WORKSPACE = (
            tuple(tensor for group in slots for tensor in group) + buffers
        )
        return out

    active_batch_size = sum(q_len > 0 for q_len in q_lens)
    if active_batch_size > 1 or _use_exact_initial_page(q_lens, kv_lens):
        stream_key = _eager_stream_key(q.device)
        out = (
            _eager_workspace(
                "batched.out",
                tuple(q.shape),
                dtype=q.dtype,
                device=q.device,
                stream_key=stream_key,
            )
            if active_q == int(q.shape[0])
            else _zero_output_like(q, active_q)
        )
        workspaces: list[torch.Tensor] = []
        workspaces.extend(
            _launch_eager_batched_prefill(
                q,
                ori_kv,
                cmp_kv,
                cmp_sparse_indices,
                ori_block_table,
                cmp_block_table,
                cu_seqlens_q,
                seqused_kv,
                sinks,
                out,
                mode=mode,
                q_lens=q_lens,
                kv_lens=kv_lens,
                cmp_ratio=int(cmp_ratio),
                softmax_scale=float(softmax_scale),
                stream_key=stream_key,
            )
        )
        _LAST_EAGER_WORKSPACE = tuple(workspaces)
        return out

    stream_key = (
        _eager_raw_stream_key(q.device)
        if (
            mode in (0, 1, 2)
            and batch_size == 1
            and q_lens[0] == 256
            and int(q.shape[0]) == 256
            and kv_lens[0] in range(256, 2049, 256)
        )
        or (
            mode in (0, 1)
            and batch_size == 1
            and q_lens[0] == 1024
            and int(q.shape[0]) == 1024
            and kv_lens[0] == 8192
        )
        else _eager_stream_key(q.device)
    )
    if active_q == int(q.shape[0]):
        out = _eager_workspace(
            "single.out",
            tuple(q.shape),
            dtype=q.dtype,
            device=q.device,
            stream_key=stream_key,
        )
    else:
        out = _zero_output_like(q, active_q)
    workspaces = []

    shared_slots: Optional[tuple[list[torch.Tensor], ...]] = None
    if mode == 2 and active_q > 0:
        worker_count = get_num_cores("cube")
        slot_cache_key: Optional[tuple[int, int, torch.dtype, int]] = None
        active_q_kv = [
            (q_len, kv_len) for q_len, kv_len in zip(q_lens, kv_lens) if q_len > 0
        ]
        if len(active_q_kv) == 1 and active_q_kv[0][0] >= 16:
            slot_cache_key = (
                stream_key[0],
                stream_key[1],
                cmp_kv.dtype,
                worker_count,
            )
            shared_slots = _MODEL_SINGLE_SCFA_SLOT_CACHE.get(slot_cache_key)
        if shared_slots is None:
            shared_slots = _allocate_scfa_pipeline_slots(
                q.device, cmp_kv.dtype, worker_count, stream_key=stream_key
            )
            if slot_cache_key is not None:
                _MODEL_SINGLE_SCFA_SLOT_CACHE[slot_cache_key] = shared_slots
        for slot_group in shared_slots:
            workspaces.extend(slot_group)

    single_q256 = (
        batch_size == 1
        and q_lens[0] == 256
        and int(q.shape[0]) == 256
        and kv_lens[0] in range(256, 2049, 256)
    )
    single_q1024_non_scfa = (
        batch_size == 1
        and mode != 2
        and q_lens[0] == 1024
        and int(q.shape[0]) == 1024
        and kv_lens[0] == 8192
    )
    identity_eager = (single_q256 and mode != 2) or single_q1024_non_scfa

    for batch_id, q_len in enumerate(q_lens):
        if q_len == 0:
            continue
        q_start, q_end = cu_q[batch_id], cu_q[batch_id + 1]
        q_view = q if identity_eager else q[q_start:q_end]
        out_view = out if identity_eager else out[q_start:q_end]
        identity_table = identity_eager
        ori_table_row = (
            ori_block_table
            if identity_table and int(ori_block_table.shape[0]) == 1
            else ori_block_table[batch_id]
        )
        if mode == 0:
            seq_workspaces = _launch_chunked_swa_sequence(
                q_view,
                ori_kv,
                ori_table_row,
                sinks,
                out_view,
                kv_len=kv_lens[batch_id],
                softmax_scale=float(softmax_scale),
                stream_key=stream_key,
            )
        elif mode == 1:
            seq_workspaces = _launch_chunked_cfa_sequence(
                q_view,
                ori_kv,
                cmp_kv,
                ori_table_row,
                (
                    cmp_block_table
                    if identity_table and int(cmp_block_table.shape[0]) == 1
                    else cmp_block_table[batch_id]
                ),
                sinks,
                out_view,
                kv_len=kv_lens[batch_id],
                softmax_scale=float(softmax_scale),
                stream_key=stream_key,
            )
        else:
            assert shared_slots is not None
            seq_workspaces = _launch_chunked_scfa_sequence(
                q_view,
                ori_kv,
                cmp_kv,
                cmp_sparse_indices[q_start:q_end],
                ori_block_table[batch_id],
                cmp_block_table[batch_id],
                sinks,
                out_view,
                shared_slots,
                kv_len=kv_lens[batch_id],
                softmax_scale=float(softmax_scale),
                stream_key=stream_key,
            )
        workspaces.extend(seq_workspaces)

    _LAST_EAGER_WORKSPACE = tuple(workspaces)
    return out


def _launch_swa_prefill(
    q: torch.Tensor,
    ori_kv: torch.Tensor,
    ori_block_table: torch.Tensor,
    sinks: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """DSV4 SWA prefill：prefix padding 后执行一次 QK 和一次 PV。"""
    total_q, num_heads_q, head_dim = q.shape
    block_n = 128
    prefix_pad = block_n - 1
    grid = (get_num_cores("cube"),)
    stage_options = _prefill_launch_options(multibuffer=False, unit_flag=True)
    stage_options["optimize_epilogue"] = False
    qk_options = dict(stage_options)
    qk_options.update(
        multibuffer=True,
        set_workspace_multibuffer=2,
        limit_auto_multi_buffer_only_for_local_buffer=False,
    )

    # padding 把每个 causal window 统一成连续 N128；padding score 会被 mask，
    # 对应 probability 为零，因此与 early 变长窗口严格等价。
    logical_k_storage = torch.zeros(
        (total_q + prefix_pad, head_dim), dtype=ori_kv.dtype, device=q.device
    )
    logical_v_storage = torch.zeros(
        (total_q + prefix_pad, head_dim), dtype=torch.float16, device=q.device
    )
    _sparse_attn_sharedkv_prefill_swa_pack_logical_k_and_fp16_v_kernel[grid](
        logical_k_storage[prefix_pad:],
        logical_v_storage[prefix_pad:],
        ori_kv,
        ori_block_table,
        TOTAL_Q=total_q,
        HEAD_DIM=head_dim,
        ORI_BLOCK_SIZE=block_n,
        D_BLOCK=256,
        **stage_options,
    )

    prob = torch.empty(
        (total_q * num_heads_q, block_n), dtype=torch.float16, device=q.device
    )
    _sparse_attn_sharedkv_prefill_swa_staged_qk_softmax_kernel[grid](
        prob,
        q,
        logical_k_storage,
        sinks,
        TOTAL_Q=total_q,
        NUM_Q_HEADS=num_heads_q,
        HEAD_DIM=head_dim,
        ORI_BLOCK_SIZE=block_n,
        M_BLOCK=64,
        N_BLOCK=block_n,
        PROB_STRIDE=block_n,
        D_BLOCK=256,
        TOKEN_START=0,
        TOKEN_END=total_q,
        SOFTMAX_SCALE=float(softmax_scale),
        FULL_WINDOW=True,
        TOKEN_BURST=16,
        QK_LOOP_UNROLL=1,
        USE_ROW_RECIPROCAL=True,
        PREFIX_PAD=prefix_pad,
        **qk_options,
    )
    out = torch.empty_like(q)
    _sparse_attn_sharedkv_prefill_swa_staged_pv_kernel[grid](
        out,
        prob,
        logical_v_storage,
        TOTAL_Q=total_q,
        NUM_Q_HEADS=num_heads_q,
        HEAD_DIM=head_dim,
        ORI_BLOCK_SIZE=block_n,
        M_BLOCK=64,
        N_BLOCK=block_n,
        PROB_STRIDE=block_n,
        D_BLOCK=512,
        TOKEN_START=0,
        TOKEN_END=total_q,
        FULL_WINDOW=True,
        TOKEN_BURST=64,
        PREFIX_PAD=prefix_pad,
        **stage_options,
    )
    _retain_eager_workspace(logical_k_storage, logical_v_storage, prob)
    return out


def _launch_cfa_prefill(
    q: torch.Tensor,
    ori_kv: torch.Tensor,
    cmp_kv: torch.Tensor,
    ori_block_table: torch.Tensor,
    cmp_block_table: torch.Tensor,
    sinks: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """DSV4 CFA prefill：original N128 与 compressed N64 staged attention。"""
    total_q, num_heads_q, head_dim = q.shape
    ori_n, cmp_n = 128, 64
    grid = (get_num_cores("cube"),)
    stage_options = _prefill_launch_options(multibuffer=False, unit_flag=True)
    stage_options["optimize_epilogue"] = False

    logical_ori_k = torch.empty(
        (total_q, head_dim), dtype=ori_kv.dtype, device=q.device
    )
    logical_ori_v = torch.empty(
        (total_q, head_dim), dtype=torch.float16, device=q.device
    )
    logical_cmp_k = torch.empty((128, head_dim), dtype=cmp_kv.dtype, device=q.device)
    logical_cmp_v = torch.empty((128, head_dim), dtype=torch.float16, device=q.device)
    ori_prob = torch.empty(
        (total_q * num_heads_q, ori_n), dtype=torch.float16, device=q.device
    )
    cmp_prob = torch.empty(
        (total_q * num_heads_q, cmp_n), dtype=torch.float16, device=q.device
    )

    for logical_k, logical_v, paged_kv, block_table, token_count in (
        (logical_ori_k, logical_ori_v, ori_kv, ori_block_table, total_q),
        (logical_cmp_k, logical_cmp_v, cmp_kv, cmp_block_table, 128),
    ):
        _sparse_attn_sharedkv_prefill_swa_pack_logical_k_and_fp16_v_kernel[grid](
            logical_k,
            logical_v,
            paged_kv,
            block_table,
            TOTAL_Q=token_count,
            HEAD_DIM=head_dim,
            ORI_BLOCK_SIZE=128,
            D_BLOCK=256,
            **stage_options,
        )

    full_start = ori_n - 1
    for token_start, token_end, full_window in (
        (0, full_start, False),
        (full_start, total_q, True),
    ):
        _sparse_attn_sharedkv_prefill_cfa_staged_qk_softmax_kernel[grid](
            ori_prob,
            cmp_prob,
            q,
            logical_ori_k,
            logical_cmp_k,
            sinks,
            TOTAL_Q=total_q,
            NUM_Q_HEADS=num_heads_q,
            HEAD_DIM=head_dim,
            ORI_BLOCK_SIZE=ori_n,
            CMP_BLOCK_SIZE=128,
            CMP_RATIO=128,
            M_BLOCK=64,
            ORI_N_BLOCK=ori_n,
            CMP_N_BLOCK=cmp_n,
            D_BLOCK=256,
            TOKEN_START=token_start,
            TOKEN_END=token_end,
            SOFTMAX_SCALE=float(softmax_scale),
            FULL_ORI_WINDOW=full_window,
            TOKEN_BURST=16,
            QK_LOOP_UNROLL=1,
            USE_ROW_RECIPROCAL=True,
            **stage_options,
        )

    out = torch.empty_like(q)
    pv_options = dict(stage_options)
    pv_options["unit_flag"] = False
    for token_start, token_end, full_window in (
        (0, full_start, False),
        (full_start, total_q, True),
    ):
        _sparse_attn_sharedkv_prefill_cfa_staged_pv_kernel[grid](
            out,
            ori_prob,
            cmp_prob,
            logical_ori_v,
            logical_cmp_v,
            TOTAL_Q=total_q,
            NUM_Q_HEADS=num_heads_q,
            HEAD_DIM=head_dim,
            ORI_BLOCK_SIZE=ori_n,
            CMP_BLOCK_SIZE=128,
            M_BLOCK=64,
            ORI_N_BLOCK=ori_n,
            CMP_N_BLOCK=cmp_n,
            D_BLOCK=512,
            TOKEN_START=token_start,
            TOKEN_END=token_end,
            FULL_ORI_WINDOW=full_window,
            TOKEN_BURST=16,
            **pv_options,
        )
    _retain_eager_workspace(
        logical_ori_k,
        logical_ori_v,
        logical_cmp_k,
        logical_cmp_v,
        ori_prob,
        cmp_prob,
    )
    return out


def _launch_scfa_prefill(
    q: torch.Tensor,
    ori_kv: torch.Tensor,
    cmp_kv: torch.Tensor,
    cmp_sparse_indices: torch.Tensor,
    ori_block_table: torch.Tensor,
    cmp_block_table: torch.Tensor,
    sinks: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """DSV4 SCFA prefill：分段 gather、QK、PV，再合并 online 状态。"""
    total_q, num_heads_q, head_dim = q.shape
    worker_count = get_num_cores("cube")
    grid = (worker_count,)
    launch_options = _prefill_launch_options(multibuffer=True, unit_flag=False)
    stage_options = dict(launch_options)
    stage_options["multibuffer"] = False
    out = torch.empty_like(q)

    total_cmp_tokens = total_q // 4
    logical_cmp_kv = torch.empty(
        (total_cmp_tokens, head_dim), dtype=cmp_kv.dtype, device=q.device
    )
    _sparse_attn_sharedkv_prefill_swa_pack_logical_kv_kernel[grid](
        logical_cmp_kv,
        cmp_kv,
        cmp_block_table,
        TOTAL_Q=total_cmp_tokens,
        HEAD_DIM=head_dim,
        ORI_BLOCK_SIZE=128,
        D_BLOCK=256,
        **launch_options,
    )

    # Original分支与SWA staged路径相同，但这里只保留未归一化的online状态。
    ori_acc = torch.empty(
        (total_q, num_heads_q, head_dim), dtype=torch.float16, device=q.device
    )
    ori_max = torch.empty((total_q, num_heads_q), dtype=torch.float32, device=q.device)
    ori_sum = torch.empty_like(ori_max)
    logical_ori_k = torch.empty(
        (total_q, head_dim), dtype=ori_kv.dtype, device=q.device
    )
    logical_ori_v = torch.empty(
        (total_q, head_dim), dtype=torch.float16, device=q.device
    )
    ori_prob = torch.empty(
        (total_q * num_heads_q, 128), dtype=torch.float16, device=q.device
    )
    _sparse_attn_sharedkv_prefill_swa_pack_logical_k_and_fp16_v_kernel[grid](
        logical_ori_k,
        logical_ori_v,
        ori_kv,
        ori_block_table,
        TOTAL_Q=total_q,
        HEAD_DIM=head_dim,
        ORI_BLOCK_SIZE=128,
        D_BLOCK=256,
        **stage_options,
    )
    for token_start, token_end, full_window, qk_d in (
        (0, 127, False, 128),
        (127, total_q, True, 256),
    ):
        _sparse_attn_sharedkv_prefill_scfa_window_qk_partial_kernel[grid](
            ori_prob,
            ori_max,
            ori_sum,
            q,
            logical_ori_k,
            TOTAL_Q=total_q,
            NUM_Q_HEADS=num_heads_q,
            HEAD_DIM=head_dim,
            ORI_BLOCK_SIZE=128,
            M_BLOCK=64,
            N_BLOCK=128,
            D_BLOCK=qk_d,
            TOKEN_START=token_start,
            TOKEN_END=token_end,
            SOFTMAX_SCALE=float(softmax_scale),
            FULL_WINDOW=full_window,
            TOKEN_BURST=16,
            **stage_options,
        )
        _sparse_attn_sharedkv_prefill_swa_staged_pv_kernel[grid](
            ori_acc,
            ori_prob,
            logical_ori_v,
            TOTAL_Q=total_q,
            NUM_Q_HEADS=num_heads_q,
            HEAD_DIM=head_dim,
            ORI_BLOCK_SIZE=128,
            M_BLOCK=64,
            N_BLOCK=128,
            PROB_STRIDE=128,
            D_BLOCK=512,
            TOKEN_START=token_start,
            TOKEN_END=token_end,
            FULL_WINDOW=full_window,
            TOKEN_BURST=16,
            **stage_options,
        )

    # 每个命名 tensor 都是独立 base pointer；四步静态展开后，编译器可证明
    # Pack/QK/softmax/PV/merge 在同一步访问不同槽，从而建立 AIC/AIV 流水。
    kv_slots = [
        torch.empty((worker_count, 512, head_dim), dtype=cmp_kv.dtype, device=q.device)
        for _ in range(4)
    ]
    valid_slots = [
        torch.empty((worker_count, 512), dtype=torch.uint8, device=q.device)
        for _ in range(4)
    ]
    score_slots = [
        torch.empty(
            (worker_count, num_heads_q, 512), dtype=torch.float32, device=q.device
        )
        for _ in range(2)
    ]
    prob_slots = [
        torch.empty(
            (worker_count, num_heads_q, 512), dtype=cmp_kv.dtype, device=q.device
        )
        for _ in range(4)
    ]
    acc_slots = [
        torch.empty(
            (worker_count, num_heads_q, head_dim), dtype=torch.float16, device=q.device
        )
        for _ in range(2)
    ]
    max_slots = [
        torch.empty((worker_count, num_heads_q), dtype=torch.float32, device=q.device)
        for _ in range(4)
    ]
    sum_slots = [torch.empty_like(slot) for slot in max_slots]

    # q<512是当前唯一保留的compact边界段；其余early和late均进入静态槽。
    first_count = 512
    first_acc = torch.empty(
        (first_count, num_heads_q, head_dim), dtype=torch.float16, device=q.device
    )
    first_max = torch.empty(
        (first_count, num_heads_q), dtype=torch.float32, device=q.device
    )
    first_sum = torch.empty_like(first_max)
    first_scores = torch.empty(
        (first_count, num_heads_q, 512), dtype=torch.float32, device=q.device
    )
    _sparse_attn_sharedkv_prefill_scfa_dense_prefix_qk_scores_kernel[grid](
        q,
        logical_cmp_kv,
        first_scores,
        first_max,
        first_sum,
        float(softmax_scale),
        Q_COUNT=first_count,
        NUM_Q_HEADS=num_heads_q,
        HEAD_DIM=head_dim,
        CMP_RATIO=4,
        MAX_CMP_TOPK=512,
        M_BLOCK=64,
        N_BLOCK=128,
        D_BLOCK=256,
        **launch_options,
    )
    _sparse_attn_sharedkv_prefill_scfa_dense_prefix_pv_partial_kernel[grid](
        logical_cmp_kv,
        first_scores,
        first_max,
        first_acc,
        Q_COUNT=first_count,
        NUM_Q_HEADS=num_heads_q,
        HEAD_DIM=head_dim,
        CMP_RATIO=4,
        MAX_CMP_TOPK=512,
        M_BLOCK=64,
        N_BLOCK=64,
        D_BLOCK=256,
        **stage_options,
    )
    _sparse_attn_sharedkv_prefill_scfa_merge_partials_kernel[grid](
        ori_acc,
        ori_max,
        ori_sum,
        first_acc,
        first_max,
        first_sum,
        sinks,
        out,
        TOTAL_Q=total_q,
        Q_TOKEN_OFFSET=0,
        Q_COUNT=first_count,
        NUM_Q_HEADS=num_heads_q,
        HEAD_DIM=head_dim,
        BLOCK_D=512,
        HEAD_BLOCK=32,
        **stage_options,
    )

    early_ranges = ((512, 1024), (1024, 1536), (1536, 2048))
    # q>=2048 是 N_LIMIT=512 的同构 late 任务；一次覆盖完整尾段，减少四级
    # 流水重复 fill/drain。该范围已在六个固定形状上验证。
    late_ranges = ((2048, total_q),)
    for q_start, q_end in early_ranges + late_ranges:
        dense_prefix = q_end <= 2048
        n_limit = ((q_end // 4 + 127) // 128) * 128 if dense_prefix else 512
        common_args = (
            ori_acc[q_start:q_end],
            ori_max[q_start:q_end],
            ori_sum[q_start:q_end],
            sinks,
            out,
            float(softmax_scale),
        )
        common_meta = {
            "TOTAL_CMP_TOKENS": total_cmp_tokens,
            "CMP_BATCH_STRIDE": total_cmp_tokens,
            "Q_TOKEN_OFFSET": q_start,
            "Q_POSITION_OFFSET": q_start,
            "Q_COUNT": q_end - q_start,
            "Q_COUNT_BUCKET": _scfa_q_count_bucket(q_end - q_start),
            "N_LIMIT": n_limit,
            "NUM_WORKERS": worker_count,
            "DENSE_PREFIX": dense_prefix,
            "BATCHED": False,
            "SCORE_GUARD": False,
        }
        _sparse_attn_sharedkv_prefill_scfa_static_slot_pipeline_kernel[grid](
            q,
            logical_cmp_kv,
            cmp_sparse_indices,
            q,
            *kv_slots,
            *valid_slots,
            *score_slots,
            *prob_slots,
            *acc_slots,
            *max_slots,
            *sum_slots,
            *common_args,
            **common_meta,
            disable_auto_inject_block_sync=True,
            **launch_options,
        )
    _retain_eager_workspace(
        logical_cmp_kv,
        logical_ori_k,
        logical_ori_v,
        ori_prob,
        ori_acc,
        ori_max,
        ori_sum,
        *kv_slots,
        *valid_slots,
        *score_slots,
        *prob_slots,
        *acc_slots,
        *max_slots,
        *sum_slots,
        first_acc,
        first_max,
        first_sum,
        first_scores,
    )
    return out


def sparse_attn_sharedkv_impl(
    q: torch.Tensor,
    *,
    ori_kv: torch.Tensor = None,
    cmp_kv: torch.Tensor = None,
    ori_sparse_indices: torch.Tensor = None,
    cmp_sparse_indices: torch.Tensor = None,
    ori_block_table: torch.Tensor = None,
    cmp_block_table: torch.Tensor = None,
    cu_seqlens_q: torch.Tensor = None,
    cu_seqlens_ori_kv: torch.Tensor = None,
    cu_seqlens_cmp_kv: torch.Tensor = None,
    seqused_q: torch.Tensor = None,
    seqused_kv: torch.Tensor = None,
    sinks: torch.Tensor = None,
    metadata: torch.Tensor = None,
    softmax_scale: float = 0,
    cmp_ratio: int = 0,
    ori_mask_mode: int = 4,
    cmp_mask_mode: int = 3,
    ori_win_left: int = 127,
    ori_win_right: int = 0,
    layout_q: str = "TND",
    layout_kv: str = "PA_ND",
    return_softmax_lse: bool = False,
):
    """仅调度当前已在六个固定形状上验证通过的 DSV4 最优路径。"""
    if return_softmax_lse:
        raise NotImplementedError(
            "Triton sparse_attn_sharedkv does not return softmax_lse."
        )
    _validate_sparse_attn_sharedkv_inputs(
        q,
        ori_kv=ori_kv,
        cmp_kv=cmp_kv,
        ori_sparse_indices=ori_sparse_indices,
        cmp_sparse_indices=cmp_sparse_indices,
        ori_block_table=ori_block_table,
        cmp_block_table=cmp_block_table,
        cu_seqlens_q=cu_seqlens_q,
        seqused_kv=seqused_kv,
        sinks=sinks,
        metadata=metadata,
        cmp_ratio=cmp_ratio,
        ori_mask_mode=ori_mask_mode,
        cmp_mask_mode=cmp_mask_mode,
        ori_win_left=ori_win_left,
        ori_win_right=ori_win_right,
        layout_q=layout_q,
        layout_kv=layout_kv,
    )
    if layout_kv != "PA_ND" or layout_q != "TND":
        raise NotImplementedError(
            "The optimized Triton paths require layout_q=TND and layout_kv=PA_ND."
        )
    if ori_sparse_indices is not None and ori_sparse_indices.numel() > 0:
        raise NotImplementedError("ori_sparse_indices is not supported.")
    if (
        ori_kv is None
        or ori_block_table is None
        or seqused_kv is None
        or metadata is None
    ):
        raise ValueError(
            "ori_kv, ori_block_table, seqused_kv and metadata are required."
        )
    if softmax_scale in (None, 0):
        softmax_scale = int(q.shape[-1]) ** -0.5

    decode_out = _try_dsv4_decode_compact_attention(
        q,
        ori_kv=ori_kv,
        cmp_kv=cmp_kv,
        cmp_sparse_indices=cmp_sparse_indices,
        ori_block_table=ori_block_table,
        cmp_block_table=cmp_block_table,
        cu_seqlens_q=cu_seqlens_q,
        seqused_kv=seqused_kv,
        sinks=sinks,
        softmax_scale=softmax_scale,
        cmp_ratio=cmp_ratio,
        ori_win_left=int(ori_win_left),
        ori_win_right=int(ori_win_right),
        layout_q=layout_q,
        layout_kv=layout_kv,
    )
    if decode_out is not None:
        return decode_out

    q, shape_info, _ = _flatten_q(q, layout_q, cu_seqlens_q)
    batch_size, max_q_len, total_q, num_heads_q, head_dim = shape_info
    if (
        batch_size != 1
        or total_q != 8192
        or max_q_len != 8192
        or num_heads_q != 64
        or head_dim != 512
        or tuple(ori_kv.shape[1:]) != (128, 1, 512)
        or int(ori_win_left) != 127
        or int(ori_win_right) != 0
        or int(seqused_kv.detach().cpu()[0].item()) != 8192
    ):
        raise NotImplementedError(
            "Only the validated DSV4 prefill shape [8192,64,512] is supported."
        )

    ori_kv = ori_kv.contiguous()
    ori_block_table = ori_block_table.contiguous()
    sinks = (
        torch.zeros((64,), dtype=torch.float32, device=q.device)
        if sinks is None or sinks.numel() == 0
        else sinks.contiguous()
    )
    has_cmp = cmp_kv is not None and cmp_kv.numel() > 0
    has_sparse = cmp_sparse_indices is not None and cmp_sparse_indices.numel() > 0
    if not has_cmp and not has_sparse and int(cmp_ratio) == 0:
        return _launch_swa_prefill(q, ori_kv, ori_block_table, sinks, softmax_scale)

    if (
        cmp_kv is None
        or cmp_block_table is None
        or tuple(cmp_kv.shape[1:]) != (128, 1, 512)
    ):
        raise NotImplementedError(
            "Compressed DSV4 paths require PA_ND cmp_kv blocks [128,1,512]."
        )
    cmp_kv = cmp_kv.contiguous()
    cmp_block_table = cmp_block_table.contiguous()
    if not has_sparse and int(cmp_ratio) == 128:
        return _launch_cfa_prefill(
            q, ori_kv, cmp_kv, ori_block_table, cmp_block_table, sinks, softmax_scale
        )
    if has_sparse and int(cmp_ratio) == 4 and int(cmp_sparse_indices.shape[-1]) == 512:
        return _launch_scfa_prefill(
            q,
            ori_kv,
            cmp_kv,
            cmp_sparse_indices.contiguous(),
            ori_block_table,
            cmp_block_table,
            sinks,
            softmax_scale,
        )
    raise NotImplementedError(
        "Input does not match SWA, CFA or SCFA DSV4 production paths."
    )


@libentry()
@triton.jit(do_not_specialize=["BATCH_SIZE"])
def _q1_scfa_direct_n64_kernel(
    q_ptr,
    kv_ptr,
    valid_ptr,
    sinks_ptr,
    cu_ptr,
    used_ptr,
    out_ptr,
    scale,
    BATCH_SIZE,
    BATCH_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    h = pid * 8 + tl.arange(0, 8)
    d = tl.arange(0, 512)
    valid_row, threshold, batch = _graphsafe_q_row_threshold(
        0, cu_ptr, used_ptr, BATCH_SIZE, BATCH_BLOCK
    )
    row_max = tl.load(sinks_ptr + h).to(tl.float32)
    row_sum = tl.full((8,), 0.0, tl.float32)
    acc = tl.zeros((8, 512), tl.float32)
    if valid_row:
        row_sum = tl.full((8,), 1.0, tl.float32)
        count = 128 + tl.minimum(threshold // 4, 512)
        q = tl.load(q_ptr + h[:, None] * 512 + d[None, :])
        n = tl.arange(0, 64)
        for start in tl.range(0, count, 64):
            token = start + n
            valid = tl.load(valid_ptr + token, mask=token < count, other=0).to(tl.int1)
            kv = tl.load(kv_ptr + token[:, None] * 512 + d[None, :])
            scores = tl.where(
                valid[None, :], tl.dot(q, tl.trans(kv)) * scale, float("-inf")
            )
            new_max = tl.maximum(row_max, tl.max(scores, 1))
            p = tl.where(valid[None, :], tl.exp(scores - new_max[:, None]), 0.0)
            alpha = tl.exp(row_max - new_max)
            row_sum = row_sum * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
            row_max = new_max
    output = tl.where(row_sum[:, None] > 0, acc / row_sum[:, None], 0.0)
    tl.store(
        out_ptr + h[:, None] * 512 + d[None, :], output.to(out_ptr.dtype.element_ty)
    )


def _launch_q1_scfa_direct_n64(
    q,
    compact_kv,
    compact_valid,
    sinks,
    cu,
    used,
    out,
    scale,
    batch,
    batch_block,
    options,
):
    global _LAST_GRAPHSAFE_WORKSPACE
    _q1_scfa_direct_n64_kernel[(8,)](
        q,
        compact_kv,
        compact_valid,
        sinks,
        cu,
        used,
        out,
        float(scale),
        batch,
        BATCH_BLOCK=batch_block,
        **options,
    )
    _LAST_GRAPHSAFE_WORKSPACE = (compact_kv, compact_valid)
    return out


_MIN_PREFILL_Q = {"swa": 512, "cfa": 128, "scfa": 128}
_SWA_Q256_KV_LENS = frozenset(range(256, 2049, 256))
# The six single-sequence shapes this operator is validated on: three decode
# cases at KV 8193 and three prefill cases at Q=KV=8192.
_VALIDATED_DECODE_KV = 8193
_VALIDATED_PREFILL_Q = 8192


def _use_validated_shapes(q_count, q_lens, kv_lens) -> bool:
    """Whether the call is one of the six validated single-sequence shapes."""
    if q_lens is None or kv_lens is None or len(q_lens) != 1 or len(kv_lens) != 1:
        return False
    q_len, kv_len = q_lens[0], kv_lens[0]
    if q_count != q_len:
        return False
    if q_len == 1:
        return kv_len == _VALIDATED_DECODE_KV
    return q_len == _VALIDATED_PREFILL_Q and kv_len == _VALIDATED_PREFILL_Q


def _use_eager_prefill(q_count, mode, q_lens, kv_lens, capturing):
    if capturing or q_lens is None:
        return False
    active = tuple(length for length in q_lens if length > 0)
    minimum = _MIN_PREFILL_Q[mode]
    swa_q256 = (
        mode == "swa"
        and q_count == 256
        and q_lens == (256,)
        and len(kv_lens) == 1
        and kv_lens[0] in _SWA_Q256_KV_LENS
    )
    batched_minimum = 256 if mode == "scfa" and max(kv_lens, default=0) > 256 else 96
    batched = len(active) > 1 and sum(active) >= batched_minimum
    per_sequence = bool(active) and all(length >= minimum for length in active)
    return swa_q256 or batched or per_sequence


def sparse_attn_sharedkv(
    q: torch.Tensor,
    *,
    host_q_lens: Optional[Sequence[int]] = None,
    host_kv_lens: Optional[Sequence[int]] = None,
    max_model_len: Optional[int] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (attention, empty_lse) for TND Q and paged shared KV.

    Provide per-sequence host lengths for uncaptured prefill; these must match
    cu_seqlens_q and seqused_kv. With no host lengths, or during NPU Graph
    capture, use the dynamic device-metadata path. max_model_len bounds every
    replay's KV length; None uses the original page table capacity.

    The six validated single-sequence shapes are dispatched to their dedicated
    schedules; every other shape uses the dynamic paths.

    Some eager outputs use per-stream scratch. Consume the result on that
    stream before the next call, or clone it in the caller if retaining it.
    The operation is inference-only and does not return softmax LSE.
    """
    if q.device.type != "npu":
        raise NotImplementedError(
            "SparseAttnSharedKV is implemented only for Ascend NPU."
        )
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("Q must be FP16 or BF16.")
    if q.dim() != 3 or tuple(q.shape[1:]) != (64, 512):
        raise ValueError("Q must have shape [T, 64, 512].")
    if kwargs.get("return_softmax_lse", False):
        raise NotImplementedError("SparseAttnSharedKV does not return softmax LSE.")
    if (
        kwargs.get("layout_q", "TND") != "TND"
        or kwargs.get("layout_kv", "PA_ND") != "PA_ND"
    ):
        raise NotImplementedError("SparseAttnSharedKV requires TND Q and PA_ND KV.")
    if (host_q_lens is None) != (host_kv_lens is None):
        raise ValueError("host_q_lens and host_kv_lens must be provided together.")
    for name, value in (("q", q), *kwargs.items()):
        if isinstance(value, torch.Tensor):
            if value.device != q.device:
                raise ValueError(f"{name} must be on the same device as Q.")
            if not value.is_contiguous():
                raise NotImplementedError(
                    f"{name} must be contiguous; no implicit copy is performed."
                )
    for name in ("ori_kv", "cmp_kv"):
        value = kwargs.get(name)
        if _present(value) and value.dtype != q.dtype:
            raise TypeError(f"{name} must have the same dtype as Q.")
    q_lens = None if host_q_lens is None else tuple(int(n) for n in host_q_lens)
    kv_lens = None if host_kv_lens is None else tuple(int(n) for n in host_kv_lens)
    if q_lens is not None:
        batch_size = kwargs["cu_seqlens_q"].numel() - 1
        if len(q_lens) != batch_size or len(kv_lens) != batch_size:
            raise ValueError("Host lengths must match the metadata batch size.")
        if any(n < 0 for n in q_lens) or sum(q_lens) > q.shape[0]:
            raise ValueError("Host query lengths are outside Q capacity.")
        if any(k < n for n, k in zip(q_lens, kv_lens)):
            raise ValueError("KV lengths must be at least query lengths.")
    if max_model_len is not None and max_model_len <= 0:
        raise ValueError("max_model_len must be positive.")
    mode = (
        "swa"
        if not _present(kwargs.get("cmp_kv"))
        else ("scfa" if _present(kwargs.get("cmp_sparse_indices")) else "cfa")
    )
    capturing = bool(torch.npu.is_current_stream_capturing())
    if not capturing and _use_validated_shapes(q.shape[0], q_lens, kv_lens):
        out = sparse_attn_sharedkv_impl(q, **kwargs)
        if isinstance(out, tuple):
            out = out[0]
    elif _use_eager_prefill(q.shape[0], mode, q_lens, kv_lens, capturing):
        out = sparse_attn_sharedkv_eager_prefill_impl(
            q,
            host_q_lens=q_lens,
            host_kv_lens=kv_lens,
            max_model_len=max_model_len,
            **kwargs,
        )
    else:
        if not capturing and kv_lens is not None:
            known_max_kv = max(kv_lens, default=0)
            max_model_len = known_max_kv
            kwargs["eager_known_max_kv"] = known_max_kv
        out = sparse_attn_sharedkv_graphsafe_impl(
            q,
            max_model_len=max_model_len,
            **kwargs,
        )
    return out, torch.empty((0,), dtype=torch.float32, device=q.device)

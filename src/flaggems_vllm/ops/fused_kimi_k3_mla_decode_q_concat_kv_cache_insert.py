# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors

import logging
from functools import partial, reduce
from operator import or_
from typing import Callable, NamedTuple

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry, libtuner

logger = logging.getLogger(__name__)
_NVIDIA = runtime.device.vendor_name == "nvidia"
_ASCEND = runtime.device.vendor_name == "ascend"
_CACHE_LAUNCH = _NVIDIA and triton.__version__.split(".")[:2] == ["3", "6"]
_launchers = {}
_plans = {}
_plans_by_numel = {}
_TensorGuards = None
if _CACHE_LAUNCH:
    from triton import knobs

    # PyTorch's compiled guard checks sizes, strides, dtype, device and tensor
    # type for fresh inputs; older builds retain the checked launch path.
    _guards = getattr(getattr(torch._C, "_dynamo", None), "guards", None)
    _TensorGuards = getattr(_guards, "TensorGuards", None)
    _current_device = getattr(torch._C, "_cuda_getDevice", None)
    _current_stream = getattr(torch._C, "_cuda_getCurrentRawStream", None)


@triton.jit
def _copy_bits(src, dst, offsets):
    # Source and destination have the same element type; no rounding is needed.
    tl.store(dst + offsets, tl.load(src + offsets))


@triton.jit
def _write_pe(src, dst, positions, table, position_offset, ROPE: tl.constexpr):
    if ROPE:
        pair = tl.arange(0, 32)
        table_row = table + tl.load(positions + position_offset) * 64
        cos = tl.load(table_row + pair)
        sin = tl.load(table_row + 32 + pair)
        lanes = tl.arange(0, 64)
        values = tl.load(src + lanes).to(tl.float32)
        even, odd = tl.split(tl.reshape(values, (32, 2)))
        rotated = tl.interleave(even * cos - odd * sin, odd * cos + even * sin)
        tl.store(dst + lanes, rotated)
    else:
        _copy_bits(src, dst, tl.arange(0, 64))


@libentry()
@libtuner(
    configs=runtime.get_tuned_config(
        "fused_kimi_k3_mla_decode_q_concat_kv_cache_insert"
    ),
    key=["HEADS", "ROPE", "SPLIT"],
)
@triton.jit
def _decode_kernel(
    QL,
    QPE,
    KV,
    KPE,
    Q,
    CACHE,
    SLOTS,
    POS,
    TABLE,
    QS0,
    QS1,
    PS0,
    PS1,
    KS0,
    CS0,
    CS1,
    SLOT_STRIDE,
    POS_STRIDE,
    HEADS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    ROPE: tl.constexpr,
    SPLIT: tl.constexpr,
    WORK_BASE,
    FLAT: tl.constexpr,
):
    WORK_PER_TOKEN: tl.constexpr = HEADS * (3 if SPLIT else 1) + 1
    if FLAT:
        linear = tl.program_id(0).to(tl.int64) + WORK_BASE
        token = linear // WORK_PER_TOKEN
        work = linear % WORK_PER_TOKEN
    else:
        token = tl.program_id(0).to(tl.int64)
        work = tl.program_id(1)
    if work == WORK_PER_TOKEN - 1:
        # The cache has exactly one owner per token, independent of head count.
        slot = tl.load(SLOTS + token * SLOT_STRIDE)
        if slot >= 0:
            dst = CACHE + (slot // PAGE_SIZE) * CS0 + (slot % PAGE_SIZE) * CS1
            _copy_bits(KV + token * 512, dst, tl.arange(0, 512))
            _write_pe(
                KPE + token * KS0, dst + 512, POS, TABLE, token * POS_STRIDE, ROPE
            )
    else:
        head = (work // 3 if SPLIT else work).to(tl.int64)
        dst = Q + (token * HEADS + head) * 576
        latent = QL + token * QS0 + head * QS1
        pe = QPE + token * PS0 + head * PS1
        if SPLIT:
            part = work % 3
            if part < 2:
                _copy_bits(latent, dst, part * 256 + tl.arange(0, 256))
            else:
                _write_pe(pe, dst + 512, POS, TABLE, token * POS_STRIDE, ROPE)
        else:
            _copy_bits(latent, dst, tl.arange(0, 512))
            _write_pe(pe, dst + 512, POS, TABLE, token * POS_STRIDE, ROPE)


if _CACHE_LAUNCH:

    @libentry()
    @libtuner(
        configs=runtime.get_tuned_config(
            "fused_kimi_k3_mla_decode_q_concat_kv_cache_insert"
        ),
        key=["HEADS", "ROPE"],
    )
    @triton.jit
    def _decode_four_heads_kernel(
        QL,
        QPE,
        KV,
        KPE,
        Q,
        CACHE,
        SLOTS,
        POS,
        TABLE,
        QS0,
        QS1,
        PS0,
        PS1,
        KS0,
        CS0,
        CS1,
        SLOT_STRIDE,
        POS_STRIDE,
        HEADS: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        ROPE: tl.constexpr,
    ):
        token = tl.program_id(0).to(tl.int64)
        group = tl.program_id(1)
        if group == HEADS // 4:
            slot = tl.load(SLOTS + token * SLOT_STRIDE)
            if slot >= 0:
                dst = CACHE + (slot // PAGE_SIZE) * CS0 + (slot % PAGE_SIZE) * CS1
                _copy_bits(KV + token * 512, dst, tl.arange(0, 512))
                _write_pe(
                    KPE + token * KS0, dst + 512, POS, TABLE, token * POS_STRIDE, ROPE
                )
        else:
            # Four complete, adjacent heads share one position and one cos/sin row.
            heads = group.to(tl.int64) * 4 + tl.arange(0, 4)
            latent = QL + token * QS0 + heads[:, None] * QS1
            pe = QPE + token * PS0 + heads[:, None] * PS1
            out = Q + (token * HEADS + heads[:, None]) * 576
            cols = tl.arange(0, 512)[None, :]
            lanes = tl.arange(0, 64)[None, :]
            latent_values = tl.load(latent + cols)
            values = tl.load(pe + lanes)
            if ROPE:
                pair = tl.arange(0, 32)
                row = TABLE + tl.load(POS + token * POS_STRIDE) * 64
                cos = tl.load(row + pair)[None, :]
                sin = tl.load(row + 32 + pair)[None, :]
                even, odd = tl.split(tl.reshape(values.to(tl.float32), (4, 32, 2)))
                values = tl.interleave(even * cos - odd * sin, odd * cos + even * sin)
            tl.store(out + cols, latent_values)
            tl.store(out + 512 + lanes, values)


def _use_split(batch, heads):
    # On H20 only a single-token batch benefits from splitting each query row.
    return batch == 1 if _NVIDIA else batch <= 64


class _CudaPlan(NamedTuple):
    page_size: int
    rope: bool
    check: Callable[..., bool]
    device: int
    kernel: object  # Owns the loaded CUDA module for the lifetime of the plan.
    run: Callable[..., None]
    metadata: tuple
    scalars: tuple


def _plan_key(query, page_size, rope):
    return (query.shape, query.stride(), query.dtype, page_size, rope)


def _try_launch(tensors, page_size, rope):
    # Numel is a cheap first lookup when decode batches alternate. Different
    # shapes, strides or dtypes can share it; only the full guard authorizes use.
    size = tensors[0].numel()
    plan = _plans_by_numel.get(size)
    if (
        plan is None
        or plan.page_size != page_size
        or plan.rope != rope
        or not plan.check(*tensors)
    ):
        other = _plans.get(_plan_key(tensors[0], page_size, rope))
        if other is None or other is plan or not other.check(*tensors):
            return False
        plan = _plans_by_numel[size] = other
    if _current_device() != plan.device:
        return False
    hooks = knobs.runtime
    if getattr(hooks.launch_enter_hook, "calls", True) or getattr(
        hooks.launch_exit_hook, "calls", True
    ):
        return False
    # TensorGuards deliberately ignores storage offset. Check fresh addresses
    # against the compiled 16-byte specialization, including views/rebindings.
    addresses = tuple(map(torch.Tensor.data_ptr, tensors))
    if reduce(or_, addresses) % _decode_kernel.divisibility:
        return False
    if not rope:
        addresses = (*addresses, addresses[6], addresses[3])
    # Match CompiledKernel's CUDA launch ABI; the original tensor-based runner
    # handles any active Triton hooks and constructs their launch metadata.
    plan.run(_current_stream(plan.device), *plan.metadata, *addresses, *plan.scalars)
    return True


def _launch(grid, pointers, scalars, kernel=_decode_kernel):
    if not _CACHE_LAUNCH:
        kernel[grid](*pointers, *scalars)
        return
    # LibEntry specializes NVIDIA tensor arguments on dtype and 16-byte
    # alignment. The checked ABI fixes all nine dtypes from the payload dtype
    # and the RoPE flag. All other specialization values and the grid are exact.
    alignment = tuple(p.data_ptr() % _decode_kernel.divisibility == 0 for p in pointers)
    first = pointers[0]
    key = (kernel, first.device, first.dtype, alignment, scalars, grid)
    entry = _launchers.get(key)
    if entry is None:
        compiled, _ = kernel[grid](*pointers, *scalars)
        if len(_launchers) >= 128:
            _launchers.clear()
        # CompiledKernel's public launch callable looks up the current stream
        # on every call. No input storage or stream is retained in the cache.
        _launchers[key] = (compiled, compiled[grid])
    else:
        compiled, launch = entry
        launch(*pointers, *scalars)
    if not (_TensorGuards and _current_stream and _current_device and all(alignment)):
        return
    # Both kernels take nine stride arguments followed by HEADS, PAGE_SIZE, ROPE.
    page_size, rope = scalars[10:12]
    tensors = pointers if rope else pointers[:7]
    if any(type(t) is not torch.Tensor for t in tensors):
        return
    try:
        check = _TensorGuards(
            *tensors, dynamic_dims_sizes=None, dynamic_dims_strides=None
        ).check
    except TypeError:
        # Other Torch builds may expose an incompatible TensorGuards constructor.
        return
    if not check(*tensors):
        # TensorGuards rejects duplicate objects; the public ABI permits them.
        return
    if len(_plans) >= 128:
        _plans.clear()
        _plans_by_numel.clear()
    plan = _CudaPlan(
        page_size,
        rope,
        check,
        first.device.index,
        compiled,
        partial(compiled.run, *grid),
        (compiled.function, compiled.packed_metadata, None, None, None),
        scalars,
    )
    _plans[_plan_key(first, page_size, rope)] = plan
    _plans_by_numel[first.numel()] = plan


def fused_kimi_k3_mla_decode_q_concat_kv_cache_insert(
    ql_nope: torch.Tensor,
    q_pe: torch.Tensor,
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    mqa_q: torch.Tensor,
    k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    cache_block_size: int,
    position_ids: torch.Tensor | None = None,
    cos_sin_cache: torch.Tensor | None = None,
) -> None:
    """Write the decode query and insert one 576-element row per valid slot.

    Nonnegative slots must be unique and in range, and positions must index the
    RoPE table. The two outputs must have disjoint storage from each other and
    from the inputs. Negative slots suppress only their cache write.
    """
    logger.debug("GEMS FUSED_KIMI_K3_MLA_DECODE_Q_CONCAT_KV_CACHE_INSERT")
    if (
        _plans_by_numel
        and type(ql_nope) is torch.Tensor
        and type(cache_block_size) is int
    ):
        if (position_ids is None) == (cos_sin_cache is None):
            tensors = (ql_nope, q_pe, kv_c_normed, k_pe, mqa_q, k_cache, slot_mapping)
            rope = position_ids is not None
            if rope:
                tensors = (*tensors, position_ids, cos_sin_cache)
            if _try_launch(tensors, cache_block_size, rope):
                return
    payloads = (ql_nope, q_pe, kv_c_normed, k_pe, mqa_q, k_cache)
    if any(not isinstance(x, torch.Tensor) for x in (*payloads, slot_mapping)):
        raise TypeError("expected tensor inputs and outputs")
    if (position_ids is None) != (cos_sin_cache is None):
        raise ValueError("position_ids and cos_sin_cache must be provided together")
    rope = position_ids is not None
    tensors = (*payloads, slot_mapping)
    if rope:
        if not isinstance(position_ids, torch.Tensor) or not isinstance(
            cos_sin_cache, torch.Tensor
        ):
            raise TypeError("position_ids and cos_sin_cache must be tensors")
        tensors = (*tensors, position_ids, cos_sin_cache)
    device, dtype = ql_nope.device, ql_nope.dtype
    if any(x.layout != torch.strided or x.device != device for x in tensors):
        raise ValueError("all tensors must be strided and on the same device")
    if dtype not in (torch.float16, torch.bfloat16) or any(
        x.dtype != dtype for x in payloads[1:]
    ):
        raise TypeError("all payload tensors must share dtype FP16 or BF16")
    if ql_nope.ndim != 3 or ql_nope.shape[2] != 512:
        raise ValueError("ql_nope must have shape [B,H,512]")
    batch, heads, _ = ql_nope.shape
    if (
        q_pe.shape != (batch, heads, 64)
        or kv_c_normed.shape != (batch, 512)
        or k_pe.shape != (batch, 64)
        or mqa_q.shape != (batch, heads, 576)
    ):
        raise ValueError("inconsistent decode tensor shapes")
    if type(cache_block_size) is not int or cache_block_size <= 0:
        raise ValueError("cache_block_size must be a positive integer")
    if k_cache.ndim != 3 or k_cache.shape[1:] != (cache_block_size, 576):
        raise ValueError("expected k_cache [num_blocks, cache_block_size, 576]")
    if any(x.stride(-1) != 1 for x in payloads):
        raise ValueError("all payload tensors require unit last-dimension stride")
    if not kv_c_normed.is_contiguous() or not mqa_q.is_contiguous():
        raise ValueError("kv_c_normed and mqa_q must be contiguous")
    if slot_mapping.dtype != torch.int64:
        raise TypeError("slot_mapping must be int64")
    if slot_mapping.shape != (batch,):
        raise ValueError("slot_mapping must have shape [B]")
    if rope:
        if position_ids.dtype != torch.int64 or cos_sin_cache.dtype != torch.float32:
            raise TypeError("RoPE requires int64 position_ids and an FP32 table")
        if position_ids.shape != (batch,):
            raise ValueError("position_ids must have shape [B]")
        if (
            cos_sin_cache.ndim != 2
            or cos_sin_cache.shape[1] != 64
            or not cos_sin_cache.is_contiguous()
            or (batch > 0 and cos_sin_cache.shape[0] == 0)
        ):
            raise ValueError("expected contiguous cos_sin_cache [max_position,64]")
    if device.type == "cpu" or device.type != runtime.device.name:
        raise NotImplementedError("an active accelerator device is required")
    if batch == 0:
        return
    split = _use_split(batch, heads)
    work_per_token = heads * (3 if split else 1) + 1
    pointers = (
        ql_nope,
        q_pe,
        kv_c_normed,
        k_pe,
        mqa_q,
        k_cache,
        slot_mapping,
        position_ids if rope else slot_mapping,
        cos_sin_cache if rope else k_pe,
    )
    scalars = (
        ql_nope.stride(0),
        ql_nope.stride(1),
        q_pe.stride(0),
        q_pe.stride(1),
        k_pe.stride(0),
        k_cache.stride(0),
        k_cache.stride(1),
        slot_mapping.stride(0),
        position_ids.stride(0) if rope else 1,
        heads,
        cache_block_size,
        rope,
        split,
    )
    with torch_device_fn.device(device):
        # Ascend limits the total grid to 65535. CUDA's limit applies to grid.y.
        # Partition the flat work domain so each cache row keeps a single owner.
        total = batch * work_per_token
        if _ASCEND and total > 65535:
            for base in range(0, total, 65535):
                count = min(total - base, 65535)
                _launch((count, 1, 1), pointers, (*scalars, base, True))
        # Four-head tiles amortize position loads once enough heads are active.
        elif (
            _CACHE_LAUNCH
            and not split
            and heads >= 32
            and heads % 4 == 0
            and batch * heads >= 4096
            and heads // 4 + 1 <= 65535
            and torch_device_fn.get_device_capability(device) == (9, 0)
        ):
            _launch(
                (batch, heads // 4 + 1, 1),
                pointers,
                scalars[:-1],
                _decode_four_heads_kernel,
            )
        elif work_per_token > 65535:
            _launch((total, 1, 1), pointers, (*scalars, 0, True))
        else:
            _launch((batch, work_per_token, 1), pointers, (*scalars, 0, False))


__all__ = ["fused_kimi_k3_mla_decode_q_concat_kv_cache_insert"]

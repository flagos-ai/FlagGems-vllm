# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors

import logging
import operator
from functools import partial, reduce
from typing import Callable, NamedTuple, Optional

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry, libtuner

logger = logging.getLogger(__name__)


@triton.jit
def _rotate(x, cos, sin, ROWS: tl.constexpr):
    even, odd = tl.split(tl.reshape(x.to(tl.float32), (ROWS, 32, 2)))
    x0 = even * cos - odd * sin
    x1 = even * sin + odd * cos
    return tl.reshape(tl.join(x0, x1), (ROWS, 64)).to(x.dtype)


@triton.jit
def _prefill_kernel(
    Q,
    K_NOPE,
    K_PE,
    LATENT,
    OUT,
    CACHE,
    SLOT,
    POS,
    COS_SIN,
    T: tl.constexpr,
    H: tl.constexpr,
    Q_S0: tl.constexpr,
    Q_S1: tl.constexpr,
    KN_S0: tl.constexpr,
    KN_S1: tl.constexpr,
    PE_S0: tl.constexpr,
    CACHE_S0: tl.constexpr,
    CACHE_S1: tl.constexpr,
    SLOT_S0: tl.constexpr,
    POS_S0: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    ROPE: tl.constexpr,
    HAS_CACHE: tl.constexpr,
    GRID_LIMIT: tl.constexpr,
    HEADS_PER_PROGRAM: tl.constexpr,
):
    head_groups: tl.constexpr = (
        (H + HEADS_PER_PROGRAM - 1) // HEADS_PER_PROGRAM if H > 0 else 1
    )
    work: tl.constexpr = T * head_groups
    # Address arithmetic is 64-bit even with a small launch and a sparse view.
    # Reuse programs beyond the backend's maximum 1-D grid.
    for batch in range((work + GRID_LIMIT - 1) // GRID_LIMIT):
        idx = tl.program_id(0).to(tl.int64) + tl.cast(batch, tl.int64) * GRID_LIMIT
        if work <= GRID_LIMIT or idx < work:
            token = idx // head_groups
            group = idx % head_groups
            pe_col = tl.arange(0, 64)
            # Preserve all 16 storage bits on the copy paths, including NaNs.
            i16 = tl.pointer_type(tl.int16)
            pe_bits = tl.load(K_PE.to(i16) + token * PE_S0 + pe_col)
            if ROPE:
                pos = tl.load(POS + token * POS_S0)
                pair = tl.arange(0, 32)
                cos = tl.load(COS_SIN + pos * 64 + pair)
                sin = tl.load(COS_SIN + pos * 64 + 32 + pair)
                pe = pe_bits.to(Q.dtype.element_ty, bitcast=True)
                pe = _rotate(pe[None, :], cos, sin, 1)
                pe_bits = tl.reshape(pe, (64,)).to(tl.int16, bitcast=True)

            if H > 0:
                head = group * HEADS_PER_PROGRAM + tl.arange(0, HEADS_PER_PROGRAM)
                active = head[:, None] < H
                kn_col = tl.arange(0, 128)
                kn_base = K_NOPE.to(i16) + token * KN_S0 + head * KN_S1
                kn = tl.load(kn_base[:, None] + kn_col[None, :], active, other=0)
                out_base = OUT.to(i16) + (token * H + head) * 192
                tl.store(out_base[:, None] + kn_col[None, :], kn, active)
                tl.store(
                    out_base[:, None] + 128 + pe_col[None, :], pe_bits[None, :], active
                )

                if ROPE:
                    q_base = Q + token * Q_S0 + head * Q_S1 + 128
                    q_tail = q_base[:, None] + pe_col[None, :]
                    x = tl.load(q_tail, active, other=0)
                    tl.store(q_tail, _rotate(x, cos, sin, HEADS_PER_PROGRAM), active)

            if HAS_CACHE and group == 0:
                slot = tl.load(SLOT + token * SLOT_S0)
                if slot >= 0:
                    row = (
                        CACHE.to(i16)
                        + (slot // PAGE_SIZE) * CACHE_S0
                        + (slot % PAGE_SIZE) * CACHE_S1
                    )
                    col = tl.arange(0, 512)
                    latent = tl.load(LATENT.to(i16) + token * 512 + col)
                    tl.store(row + col, latent)
                    tl.store(row + 512 + pe_col, pe_bits)


# Copy-only trials can overwrite the same rows. A RoPE trial would rotate Q
# again, so the stateful path uses one fixed launch with no tuning replay.
_fixed_kernel = libentry()(_prefill_kernel)
if runtime.device.vendor_name == "nvidia":
    _copy_kernel = libentry()(
        libtuner(
            configs=runtime.get_tuned_config(
                "fused_kimi_k3_mla_key_concat_kv_cache_insert"
            ),
            key=["T", "H", "KN_S0", "KN_S1", "PE_S0", "CACHE_S0", "CACHE_S1"],
        )(_prefill_kernel)
    )


# The specialized launcher below is tested with the H20 Torch/FlagTree ABI.
# Other builds and devices retain the normal checked LibEntry launch.
_cuda_plans = {}
_tensor_guards = None
if (
    runtime.device.vendor_name == "nvidia"
    and torch.__version__.split(".")[:2] == ["2", "13"]
    and triton.__version__.split(".")[:2] == ["3", "6"]
):
    _knobs = getattr(triton, "knobs", None)
    _cuda_hooks = getattr(_knobs, "runtime", None)
    _cuda_device = getattr(torch._C, "_cuda_getDevice", None)
    _cuda_stream = getattr(torch._C, "_cuda_getCurrentRawStream", None)
    if _cuda_hooks is not None and _cuda_device and _cuda_stream:
        _guards = getattr(getattr(torch._C, "_dynamo", None), "guards", None)
        _tensor_guards = getattr(_guards, "TensorGuards", None)


class _CudaLaunch(NamedTuple):
    check: Callable
    device: int
    alignment: int
    tensor_types: tuple
    kernel: object  # Owns the loaded CUDA module. No input tensor is retained.
    run: Callable
    metadata: tuple
    constants: tuple


def _launch_hooks_active():
    return getattr(
        getattr(_cuda_hooks, "launch_enter_hook", None), "calls", True
    ) or getattr(getattr(_cuda_hooks, "launch_exit_hook", None), "calls", True)


def _remember_cuda_launch(entry, result, pointers, constants, config, grid, page, rope):
    # LibEntry specializes NVIDIA pointers on dtype and alignment. TensorGuards
    # checks shape, strides, dtype, device and dispatch keys on every rebind.
    tensors = pointers if rope else pointers[:7]
    if (
        entry._has_flagtune_tuner
        or _launch_hooks_active()
        or any(type(t) is not torch.Tensor for t in tensors)
        or any(t.data_ptr() % entry.divisibility for t in tensors)
        or "H20" not in torch.cuda.get_device_name(tensors[0].device)
        or getattr(entry.fn, "shared_config_pre_hook", None)
        or getattr(getattr(entry.fn, "best_config", None), "pre_hook", None)
    ):
        return
    try:
        check = _tensor_guards(
            *tensors, dynamic_dims_sizes=None, dynamic_dims_strides=None
        ).check
    except TypeError:
        # Some downstream Torch builds expose a different guard constructor.
        return
    if not check(*tensors):
        return
    compiled, selected = result
    if not all(
        hasattr(compiled, attr) for attr in ("run", "function", "packed_metadata")
    ):
        return
    tile = {**config, **selected}["HEADS_PER_PROGRAM"]
    dims = (*grid({"HEADS_PER_PROGRAM": tile}), 1, 1)
    plan = _CudaLaunch(
        check,
        tensors[0].device.index,
        entry.divisibility,
        tuple(map(type, tensors)),
        compiled,
        partial(compiled.run, *dims),
        (compiled.function, compiled.packed_metadata, None, None, None),
        (*constants, tile),
    )
    if len(_cuda_plans) >= 64:
        _cuda_plans.clear()
    _cuda_plans[tensors[0].numel(), page, rope] = plan


def _try_cuda_launch(tensors, page, rope):
    plan = _cuda_plans.get((tensors[0].numel(), page, rope))
    if (
        plan is None
        or tuple(map(type, tensors)) != plan.tensor_types
        or _cuda_device() != plan.device
        or _launch_hooks_active()
        or not plan.check(*tensors)
    ):
        return False
    # A view can pass all metadata guards with a different pointer alignment.
    addresses = tuple(map(torch.Tensor.data_ptr, tensors))
    if reduce(operator.or_, addresses) % plan.alignment:
        return False
    if not rope:
        addresses = (*addresses, addresses[6], addresses[2])
    plan.run(_cuda_stream(plan.device), *plan.metadata, *addresses, *plan.constants)
    return True


def _launch(
    q,
    k_nope,
    k_pe,
    kv_c_normed,
    k_out,
    k_cache,
    slot_mapping,
    cache_block_size,
    position_ids=None,
    cos_sin_cache=None,
    *,
    heads_per_program=None,
):
    tokens, heads, _ = q.shape
    rope = position_ids is not None
    grid_limit = 65535 if runtime.device.vendor_name == "ascend" else 2**31 - 1

    def grid(meta):
        return (
            min(
                tokens * max(1, triton.cdiv(heads, meta["HEADS_PER_PROGRAM"])),
                grid_limit,
            ),
        )

    config = {}
    if rope or runtime.device.vendor_name != "nvidia" or heads_per_program is not None:
        kernel = _fixed_kernel
        # On H20, eight heads per program reduce the 96/128-head RoPE
        # kernel time; the smaller projections favor groups of four.
        default_heads = (
            8 if runtime.device.vendor_name == "nvidia" and heads >= 96 else 4
        )
        config["HEADS_PER_PROGRAM"] = heads_per_program or default_heads
    else:
        kernel = _copy_kernel
    q_ptr = q if heads else k_pe
    if runtime.device.vendor_name == "ascend":
        # BiSheng requires the global-memory element type at the kernel boundary.
        # Equal-size views preserve the packed projection strides and offsets.
        k_nope = k_nope.view(torch.int16)
        k_pe = k_pe.view(torch.int16)
        kv_c_normed = kv_c_normed.view(torch.int16)
        k_out = k_out.view(torch.int16)
        k_cache = k_cache.view(torch.int16)
    pointers = (
        q_ptr,
        k_nope if heads else k_pe,
        k_pe,
        kv_c_normed,
        k_out if heads else k_pe,
        k_cache if k_cache.size(0) else k_pe,
        slot_mapping,
        position_ids if rope else slot_mapping,
        cos_sin_cache if rope else k_pe,
    )
    constants = (
        tokens,
        heads,
        q.stride(0),
        q.stride(1),
        k_nope.stride(0),
        k_nope.stride(1),
        k_pe.stride(0),
        k_cache.stride(0),
        k_cache.stride(1),
        slot_mapping.stride(0),
        position_ids.stride(0) if rope else 1,
        cache_block_size,
        rope,
        k_cache.size(0) != 0,
        grid_limit,
    )
    with torch_device_fn.device(q.device):
        result = kernel[grid](*pointers, *constants, **config)
        if (
            _tensor_guards is not None
            and heads
            and k_cache.size(0)
            and heads_per_program is None
        ):
            _remember_cuda_launch(
                kernel,
                result,
                pointers,
                constants,
                config,
                grid,
                cache_block_size,
                rope,
            )


def fused_kimi_k3_mla_key_concat_kv_cache_insert(
    q: torch.Tensor,
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    kv_c_normed: torch.Tensor,
    k_out: torch.Tensor,
    k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    cache_block_size: int,
    position_ids: Optional[torch.Tensor] = None,
    cos_sin_cache: Optional[torch.Tensor] = None,
) -> None:
    """Write the prefill full key and insert the latent into the mapped pages.

    With positions, rotate adjacent pairs in each Q tail in place and share
    the rotated K tail across the heads and cache entry. Negative slots skip
    only the cache write. Nonnegative slots must be unique and in range, and
    positions must index the cosine/sine table. Written elements must be
    disjoint from other tensor arguments, including packed projection views.
    """
    logger.debug("GEMS FUSED_KIMI_K3_MLA_KEY_CONCAT_KV_CACHE_INSERT")
    if (position_ids is None) != (cos_sin_cache is None):
        raise ValueError("position_ids and cos_sin_cache must be provided together")
    data = (q, k_nope, k_pe, kv_c_normed, k_out, k_cache)
    tensors = (*data, slot_mapping)
    if position_ids is not None:
        tensors += (position_ids, cos_sin_cache)
    if (
        _cuda_plans
        and type(q) is torch.Tensor
        and type(cache_block_size) is int
        and _try_cuda_launch(tensors, cache_block_size, position_ids is not None)
    ):
        return
    if any(not isinstance(t, torch.Tensor) for t in tensors):
        raise TypeError("expected tensor arguments")
    if any(t.layout != torch.strided for t in tensors):
        raise ValueError("all tensors must have strided layout")
    if any(t.device != q.device for t in tensors):
        raise ValueError("all tensors must be on the same device")
    if q.dtype not in (torch.bfloat16, torch.float16) or any(
        t.dtype != q.dtype for t in data[1:]
    ):
        raise TypeError(
            "q, k_nope, k_pe, kv_c_normed, k_out and k_cache must share BF16 or FP16 dtype"
        )
    if q.ndim != 3 or q.size(2) != 192:
        raise ValueError("q must have shape [T, H, 192]")
    tokens, heads, _ = q.shape
    if k_nope.shape != (tokens, heads, 128) or k_pe.shape != (tokens, 64):
        raise ValueError("expected k_nope [T, H, 128] and k_pe [T, 64]")
    if kv_c_normed.shape != (tokens, 512) or k_out.shape != q.shape:
        raise ValueError("expected kv_c_normed [T, 512] and k_out [T, H, 192]")
    cache_block_size = operator.index(cache_block_size)
    if (
        cache_block_size <= 0
        or k_cache.ndim != 3
        or k_cache.shape[1:] != (cache_block_size, 576)
    ):
        raise ValueError(
            "k_cache must have shape [P, cache_block_size, 576] with a positive block size"
        )
    if any(t.stride(-1) != 1 for t in (q, k_nope, k_pe, k_cache)):
        raise ValueError(
            "q, k_nope, k_pe and k_cache must have unit last-dimension stride"
        )
    if not kv_c_normed.is_contiguous() or not k_out.is_contiguous():
        raise ValueError("kv_c_normed and k_out must be contiguous")
    if slot_mapping.dtype != torch.int64 or slot_mapping.shape != (tokens,):
        raise ValueError("slot_mapping must be int64 [T]")
    if position_ids is not None:
        if position_ids.dtype != torch.int64 or position_ids.shape != (tokens,):
            raise ValueError("position_ids must be int64 [T]")
        if (
            cos_sin_cache.dtype != torch.float32
            or cos_sin_cache.ndim != 2
            or cos_sin_cache.size(1) != 64
            or not cos_sin_cache.is_contiguous()
        ):
            raise ValueError("cos_sin_cache must be contiguous FP32 [max_position, 64]")
        if tokens and cos_sin_cache.size(0) == 0:
            raise ValueError("nonempty position_ids require a nonempty cos_sin_cache")
    if q.device.type == "cpu" or q.device.type != runtime.device.name:
        raise NotImplementedError("an active accelerator device is required")
    if tokens:
        _launch(
            q,
            k_nope,
            k_pe,
            kv_c_normed,
            k_out,
            k_cache,
            slot_mapping,
            cache_block_size,
            position_ids,
            cos_sin_cache,
        )


__all__ = ["fused_kimi_k3_mla_key_concat_kv_cache_insert"]

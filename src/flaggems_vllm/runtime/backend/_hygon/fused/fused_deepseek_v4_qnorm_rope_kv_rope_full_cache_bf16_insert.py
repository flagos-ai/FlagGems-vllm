# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

import logging
import os
import sys
from typing import Callable, NamedTuple

import torch
import triton

from flaggems_vllm.ops.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert import (
    _check_args,
    _full_cache_kernel,
    _large_qnorm_full_cache_kernel,
)

try:
    from triton import knobs
except ImportError:
    knobs = None

logger = logging.getLogger(__name__)
_generic = _full_cache_kernel.jit_function
_flat = _large_qnorm_full_cache_kernel.jit_function
_runtime = getattr(knobs, "runtime", None)
_compilation = getattr(knobs, "compilation", None)
_hcu = getattr(knobs, "hcu", None)
_hook_chain = getattr(knobs, "HookChain", None)
_tensor_guard = getattr(
    getattr(getattr(torch._C, "_dynamo", None), "guards", None), "TensorGuards", None
)
_kernel = _generic
if triton.__version__.split(".")[:2] == ["3", "6"]:
    try:
        # Small decode tiles take the same time without pointer specialization.
        # A single binary then handles every alignment and backing storage size.
        _pointers = _generic.arg_names[:6]
        _kernel = triton.jit(
            _generic.fn,
            do_not_specialize=_pointers,
            do_not_specialize_on_alignment=_pointers,
        )
    except (AttributeError, TypeError):
        pass


class _Launch(NamedTuple):
    options: tuple
    matches: Callable
    compact: bool
    device: int
    scalars: tuple
    run: Callable


_last = None


def _idle(hook):
    return hook is None or (type(hook) is _hook_chain and not hook.calls)


def _can_reuse():
    prism = sys.modules.get("flagtree._flagprism")
    return (
        _kernel is not _generic
        and _tensor_guard is not None
        and not getattr(_runtime, "debug", True)
        and not getattr(_runtime, "interpret", True)
        and not getattr(_compilation, "instrumentation_mode", True)
        and not getattr(_runtime, "jit_cache_hook", True)
        and not getattr(_runtime, "jit_post_compile_hook", True)
        and not getattr(_runtime, "add_stages_inspection_hook", True)
        and _idle(getattr(_runtime, "launch_enter_hook", True))
        and _idle(getattr(_runtime, "launch_exit_hook", True))
        and getattr(_hcu, "use_buffer_ops", False)
        and not any(
            getattr(j, "debug", True) or getattr(j, "pre_run_hooks", True)
            for j in (_generic, _kernel, _flat)
        )
        and (prism is None or not getattr(prism, "_components", True))
        and not os.environ.get("FLAGTREE_LITE_DIST")
    )


def _compact(tensors):
    # HCU's D/S specialization describes the address and the whole storage.
    return all(
        type(t) is torch.Tensor
        and not hasattr(t, "ptr_range")
        and t.data_ptr() % 16 == 0
        and t.untyped_storage().nbytes() < 2**31
        for t in tensors
    )


def _known_pointers(compiled, compact):
    attrs = getattr(getattr(compiled, "src", None), "attrs", None)
    allowed = (("tt.divisibility", 16), ("tt.pointer_range", 32)) if compact else ()
    return isinstance(attrs, dict) and all(
        isinstance(index, tuple)
        and len(index) == 1
        and (index[0] >= 6 or all(tuple(attr) in allowed for attr in values))
        for index, values in attrs.items()
    )


def fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert(
    q: torch.Tensor,
    kv: torch.Tensor,
    k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    position_ids: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
    cache_block_size: int,
    apply_q_norm: bool = True,
) -> None:
    """Launch BF16 full-cache updates on the caller's current HCU stream."""
    global _last
    logger.debug(
        "GEMS_HYGON FUSED_DEEPSEEK_V4_QNORM_ROPE_KV_ROPE_FULL_CACHE_BF16_INSERT"
    )
    tensors = (q, kv, k_cache, slot_mapping, position_ids, cos_sin_cache)
    options = (float(eps), cache_block_size, bool(apply_q_norm))
    reuse = type(cache_block_size) is int and _can_reuse()
    plan = _last
    if (
        reuse
        and plan is not None
        and options == plan.options
        and plan.matches(*tensors)
        and torch.cuda.current_device() == plan.device
        and (not plan.compact or _compact(tensors))
    ):
        plan.run(*tensors, *plan.scalars)
        return

    _check_args(*tensors, cache_block_size)
    n, h, _ = q.shape
    rows = n * h
    inserts = slot_mapping.numel()
    compact = options[2] and rows >= 4096 and _compact(tensors)
    if compact:
        kernel = _flat
        qt, count, mode, warps = triton.cdiv(rows, 8), rows, rows >= 2**31 - 8, 8
    else:
        kernel = (
            _generic
            if getattr(_generic, "pre_run_hooks", ())
            or getattr(_generic, "debug", None)
            else _kernel
        )
        qt = n * triton.cdiv(h, 8)
        count, mode, warps = qt, options[2], 4
    grid = (qt + triton.cdiv(inserts, 8), 1, 1)
    if not grid[0]:
        return
    scalars = (
        options[0],
        count,
        inserts,
        *k_cache.stride()[:2],
        h,
        cache_block_size,
        mode,
    )
    with torch.cuda.device(q.device):
        if not reuse:
            kernel[grid](*tensors, *scalars, num_warps=warps)
            return
        # A miss launches exactly once on real tensors before publishing the runner.
        with _full_cache_kernel.lock:
            compiled = kernel[grid](*tensors, *scalars, num_warps=warps)
        if not _known_pointers(compiled, compact):
            return
        try:
            guard = _tensor_guard(
                *tensors,
                dynamic_dims_sizes=[list(t.shape) for t in tensors],
                dynamic_dims_strides=[list(t.stride()) for t in tensors],
            )
        except TypeError:
            return
        _last = _Launch(
            options, guard.check, compact, q.get_device(), scalars, compiled[grid]
        )

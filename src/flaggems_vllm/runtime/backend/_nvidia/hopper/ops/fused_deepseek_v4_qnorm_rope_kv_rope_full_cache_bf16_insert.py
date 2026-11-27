# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

import logging
import threading
from collections import OrderedDict
from functools import partial

import torch
import triton
from triton import knobs
from triton.runtime import driver

from flaggems_vllm.ops.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert import (
    _check_args,
    _full_cache_kernel,
    _large_qnorm_full_cache_kernel,
)

logger = logging.getLogger(__name__)
_cache_lock = threading.Lock()
_plans = OrderedDict()
_CACHE_LIMIT = 128
_tensor_guard = getattr(
    getattr(getattr(torch._C, "_dynamo", None), "guards", None), "TensorGuards", None
)
_hook_chain_type = getattr(knobs, "HookChain", None)
_last_plan = None


def _empty_hook(hook):
    return hook is None or (type(hook) is _hook_chain_type and not hook.calls)


class _LaunchPlan:
    def __init__(self, tensors, block_size, norm, eps):
        q, _, cache, slots, _, _ = tensors
        n, h, _ = q.shape
        self.device = q.get_device()
        self.options = (block_size, norm, eps)
        i = slots.numel()
        rows = n * h
        page_stride, row_stride, _ = cache.stride()
        # A flat group of eight heads reduces inactive lanes and avoids
        # reloading the rotary tail once enough normalized Q rows are present.
        if norm and rows >= 1024:
            self.kernel = _large_qnorm_full_cache_kernel
            query_tiles = triton.cdiv(rows, 8)
            self.scalars = (
                eps,
                rows,
                i,
                page_stride,
                row_stride,
                h,
                block_size,
                rows >= 2**31 - 8,
            )
        else:
            self.kernel = _full_cache_kernel
            query_tiles = n * triton.cdiv(h, 8)
            self.scalars = (
                eps,
                query_tiles,
                i,
                page_stride,
                row_stride,
                h,
                block_size,
                norm,
            )
        self.grid = (query_tiles + triton.cdiv(i, 8), 1, 1)
        dtypes = (torch.bfloat16,) * 3 + (torch.int64, torch.int64, torch.float32)
        with torch.cuda.device(self.device):
            compiled = self.kernel.jit_function.warmup(
                *dtypes, *self.scalars, grid=self.grid, num_warps=4
            )
            self.run = compiled[self.grid]
        launcher = compiled.run
        self.direct = None
        if (
            getattr(launcher, "global_scratch_size", None) == 0
            and getattr(launcher, "profile_scratch_size", None) == 0
            and getattr(launcher, "launch_cooperative_grid", None) is False
            and getattr(launcher, "launch_pdl", None) is False
        ):
            # This module targets Triton 3.6's CUDA launcher. A plain grid
            # requires neither scratch allocation nor launch metadata.
            self.direct = partial(launcher.launch, *self.grid)
            self.prefix = (
                compiled.function,
                False,
                False,
                None,
                None,
                compiled.packed_metadata,
                None,
                None,
                None,
            )
        self.stream = driver.active.get_current_stream
        self.matches = None
        if _tensor_guard is not None:
            # Snapshot exact metadata, including singleton dimensions. Keep
            # addresses and streams out of the plan so new views can reuse it.
            try:
                guard = _tensor_guard(
                    *tensors,
                    dynamic_dims_sizes=[list(x.shape) for x in tensors],
                    dynamic_dims_strides=[list(x.stride()) for x in tensors],
                )
            except TypeError:
                # Older PyTorch versions can expose a different guard API.
                return
            self.matches = guard.check

    def launch(self, q, kv, cache, slots, positions, table):
        q_ptr, kv_ptr, cache_ptr = q.data_ptr(), kv.data_ptr(), cache.data_ptr()
        slots_ptr, positions_ptr, table_ptr = (
            slots.data_ptr(),
            positions.data_ptr(),
            table.data_ptr(),
        )
        if (q_ptr | kv_ptr | cache_ptr | slots_ptr | positions_ptr | table_ptr) & 15:
            self.kernel[self.grid](
                q, kv, cache, slots, positions, table, *self.scalars, num_warps=4
            )
        elif (
            self.direct is not None
            and _empty_hook(knobs.runtime.launch_enter_hook)
            and _empty_hook(knobs.runtime.launch_exit_hook)
        ):
            self.direct(
                self.stream(self.device),
                *self.prefix,
                q_ptr,
                kv_ptr,
                cache_ptr,
                slots_ptr,
                positions_ptr,
                table_ptr,
                *self.scalars,
            )
        else:
            self.run(
                q,
                kv,
                cache,
                slots,
                positions,
                table,
                *self.scalars,
                stream=self.stream(self.device),
            )


def _prepare(tensors, block_size, norm, eps):
    q, _, cache, slots, _, _ = tensors
    if not all(isinstance(t, torch.Tensor) for t in tensors) or not isinstance(
        block_size, int
    ):
        _check_args(*tensors, block_size)
    key = (q.device, q.shape, slots.shape, cache.stride(), block_size, norm, eps)
    with _cache_lock:
        plan = _plans.get(key)
        if plan is None or plan.matches is None or not plan.matches(*tensors):
            _check_args(*tensors, block_size)
            if not q.is_cuda:
                raise NotImplementedError(
                    "the NVIDIA full-cache implementation requires CUDA tensors"
                )
            if q.numel() == 0 and slots.numel() == 0:
                return None
            if plan is None or plan.matches is not None:
                plan = _LaunchPlan(tensors, block_size, norm, eps)
                _plans[key] = plan
                if len(_plans) > _CACHE_LIMIT:
                    _plans.popitem(last=False)
        _plans.move_to_end(key)
        return plan


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
    """Validate the current tensors and launch on the calling thread's CUDA stream."""
    logger.debug(
        "GEMS_NVIDIA FUSED_DEEPSEEK_V4_QNORM_ROPE_KV_ROPE_FULL_CACHE_BF16_INSERT"
    )
    global _last_plan
    tensors = (q, kv, k_cache, slot_mapping, position_ids, cos_sin_cache)
    norm, eps = bool(apply_q_norm), float(eps)
    plan = _last_plan
    if (
        plan is None
        or type(cache_block_size) is not int
        or (cache_block_size, norm, eps) != plan.options
        or plan.matches is None
        or not plan.matches(*tensors)
    ):
        plan = _prepare(tensors, cache_block_size, norm, eps)
        if plan is None:
            return
        _last_plan = plan
    if torch.cuda.current_device() == plan.device:
        plan.launch(*tensors)
    else:
        with torch.cuda.device(plan.device):
            plan.launch(*tensors)

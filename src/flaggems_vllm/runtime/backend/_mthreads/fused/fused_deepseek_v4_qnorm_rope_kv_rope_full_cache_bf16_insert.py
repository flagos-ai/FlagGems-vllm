# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

import logging
import threading

import torch
import triton
from triton.runtime import driver

from flaggems_vllm.ops.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert import (
    _check_args,
    _full_cache_kernel,
    _large_qnorm_full_cache_kernel,
)
from flaggems_vllm.runtime import torch_device_fn

knobs = getattr(triton, "knobs", None)
try:
    from flagtree import _flagprism
except ImportError:
    _flagprism = None

# A registered debugger or profiler can change its options between calls.
# With an unknown registry API, always delegate to FlagTree's JIT.
_prism_components = (
    getattr(_flagprism, "_registered_components", lambda: True)
    if _flagprism is not None
    else lambda: False
)

try:
    from torch_musa._MUSAC import _musa_getCurrentRawStream as _get_stream
except ImportError:

    def _get_stream(device):
        return driver.active.get_current_stream(device)


logger = logging.getLogger(__name__)
_state = threading.local()
_tensor_guard = getattr(
    getattr(getattr(torch._C, "_dynamo", None), "guards", None), "TensorGuards", None
)
_hook_chain = getattr(knobs, "HookChain", None)


def _needs_jit(kernel):
    # These switches affect compilation as well as the launch. LibEntry's
    # cached specialization cannot replay a JIT pre-hook or instrument a binary.
    return (
        kernel.jit_function.pre_run_hooks
        or knobs.runtime.debug
        or knobs.compilation.instrumentation_mode
        or knobs.runtime.add_stages_inspection_hook
        or _prism_components()
    )


def _no_hook(hook):
    return hook is None or (type(hook) is _hook_chain and not hook.calls)


def _aligned(pointers):
    q, kv, cache, slots, positions, table = pointers
    return (q | kv | cache | slots | positions | table) & 15 == 0


class _Launch:
    def __init__(self, kernel, grid, scalars, compiled, guard, options, device, warps):
        self.kernel = kernel
        self.grid = grid
        self.scalars = scalars
        self.run = compiled[grid]
        self.matches = guard.check
        self.options = options
        self.device = device
        self.warps = warps
        self.plain_launcher = None
        # FlagTree's MUSA 3.6 launcher accepts metadata and hooks explicitly.
        # The wrapper itself owns scratch allocation and buffer lifetimes.
        if (
            getattr(compiled.metadata, "backend_name", None) == "musa"
            and getattr(compiled.metadata, "triton_version", None) == "3.6.0"
            and getattr(compiled.run, "_debug_enabled", None) is False
            and getattr(compiled.run, "_needs_runtime_expansion", None) is False
        ):
            self.plain_launcher = compiled.run
            self.launch_args = (
                compiled.function,
                compiled.packed_metadata,
                None,
                None,
                None,
            )

    def __call__(self, tensors):
        if _needs_jit(self.kernel):
            self.kernel.jit_function[self.grid](
                *tensors, *self.scalars, num_warps=self.warps
            )
            return
        pointers = tuple(t.data_ptr() for t in tensors)
        if not _aligned(pointers):
            self.kernel[self.grid](*tensors, *self.scalars, num_warps=self.warps)
            return
        # The MUSA launcher accepts device addresses directly. Instrumented
        # launches keep tensors for launch metadata and the FlagPrism debugger.
        stream = _get_stream(self.device)
        if (
            self.plain_launcher is not None
            and _no_hook(knobs.runtime.launch_enter_hook)
            and _no_hook(knobs.runtime.launch_exit_hook)
        ):
            self.plain_launcher(
                *self.grid, stream, *self.launch_args, *pointers, *self.scalars
            )
            return
        self.run(*tensors, *self.scalars, stream=stream)


def _first_launch(tensors, options):
    block_size, norm, eps = options
    _check_args(*tensors, block_size)
    q, _, cache, slots, _, _ = tensors
    if q.device.type != "musa":
        raise NotImplementedError("the MThreads implementation requires MUSA tensors")
    n, heads, _ = q.shape
    inserts = slots.numel()
    rows = n * heads
    page_stride, row_stride, _ = cache.stride()
    if norm and rows >= 4096:
        # On the S5000 each of eight warps reduces one complete Q row.
        kernel = _large_qnorm_full_cache_kernel
        warps = 8
        query_tiles = triton.cdiv(rows, 8)
        scalars = (
            eps,
            rows,
            inserts,
            page_stride,
            row_stride,
            heads,
            block_size,
            rows >= 2**31 - 8,
        )
    else:
        kernel = _full_cache_kernel
        warps = 4
        query_tiles = n * triton.cdiv(heads, 8)
        scalars = (
            eps,
            query_tiles,
            inserts,
            page_stride,
            row_stride,
            heads,
            block_size,
            norm,
        )
    grid = (query_tiles + triton.cdiv(inserts, 8), 1, 1)
    if not grid[0]:
        return
    device = q.get_device()
    with torch_device_fn.device(device):
        if knobs is not None and _needs_jit(kernel):
            kernel.jit_function[grid](*tensors, *scalars, num_warps=warps)
            return
        compiled, _ = kernel[grid](*tensors, *scalars, num_warps=warps)
        # Cache metadata for one recent shape per thread. No tensor, address,
        # stream, or tensor identity is retained across calls.
        if (
            knobs is None
            or _tensor_guard is None
            or any(type(t) is not torch.Tensor for t in tensors)
        ):
            return
        if not _aligned(tuple(t.data_ptr() for t in tensors)):
            return
        try:
            guard = _tensor_guard(
                *tensors,
                dynamic_dims_sizes=[list(t.shape) for t in tensors],
                dynamic_dims_strides=[list(t.stride()) for t in tensors],
            )
        except TypeError:
            return
        _state.launch = _Launch(
            kernel, grid, scalars, compiled, guard, options, device, warps
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
    """Update Q and the BF16 cache on the caller's current MUSA stream."""
    logger.debug(
        "GEMS_MTHREADS FUSED_DEEPSEEK_V4_QNORM_ROPE_KV_ROPE_FULL_CACHE_BF16_INSERT"
    )
    tensors = (q, kv, k_cache, slot_mapping, position_ids, cos_sin_cache)
    options = (cache_block_size, bool(apply_q_norm), float(eps))
    launch = getattr(_state, "launch", None)
    if (
        launch is None
        or type(cache_block_size) is not int
        or options != launch.options
        or not launch.matches(*tensors)
    ):
        _first_launch(tensors, options)
    elif torch_device_fn.current_device() == launch.device:
        launch(tensors)
    else:
        with torch_device_fn.device(launch.device):
            launch(tensors)

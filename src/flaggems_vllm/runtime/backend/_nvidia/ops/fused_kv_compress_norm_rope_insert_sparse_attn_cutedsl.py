# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""NVIDIA autotuning and checked CUDA launch reuse for overlapping C4."""

from __future__ import annotations

import logging
from functools import reduce
from operator import or_

import torch
import triton

from flaggems_vllm import runtime
from flaggems_vllm.ops.fused_kv_compress_norm_rope_insert_sparse_attn_cutedsl import (
    _dispatch_c4,
    _fused_c4_kernel,
)
from flaggems_vllm.utils import libentry, libtuner

logger = logging.getLogger(__name__)
_OP = "fused_kv_compress_norm_rope_insert_sparse_attn_cutedsl"

_fused_c4 = libentry()(
    libtuner(
        configs=runtime.get_tuned_config(_OP),
        key=["N", "FULL", "FP8"],
        warmup=5,
        rep=15,
        use_cuda_graph=False,
    )(_fused_c4_kernel)
)

# Only reuse a launcher for an already validated CUDA geometry. TensorGuards
# compares tensor metadata in C++ and does not retain the caller's tensors.
_last_cuda_plan = None
_guard_factory = getattr(
    getattr(getattr(torch._C, "_dynamo", None), "guards", None),
    "TensorGuards",
    None,
)
_hooks = getattr(getattr(triton, "knobs", None), "runtime", None)


def _hooks_clear():
    # Unknown hook APIs use ordinary Triton dispatch, with tensor metadata.
    return not (
        getattr(getattr(_hooks, "launch_enter_hook", None), "calls", True)
        or getattr(getattr(_hooks, "launch_exit_hook", None), "calls", True)
    )


class _CudaPlan:
    def __init__(self, tensors, options, compiled, grid, args):
        self.check = _guard_factory(
            *tensors,
            dynamic_dims_sizes=[list(t.shape) for t in tensors],
            dynamic_dims_strides=[list(t.stride()) for t in tensors],
        ).check
        self.options = options
        self.device = tensors[0].device.index
        self.launch = compiled[(grid, 1, 1)]
        self.args = args

    def try_launch(self, tensors, options):
        if (
            options != self.options
            or not _hooks_clear()
            or not self.check(*tensors)
            or torch.cuda.current_device() != self.device
        ):
            return False
        pointers = tuple(map(torch.Tensor.data_ptr, tensors))
        if reduce(or_, pointers) % _fused_c4.divisibility:
            return False
        if len(pointers) == 9:
            pointers += (pointers[0],)
        # CompiledKernel's grid launcher resolves the current stream per call.
        self.launch(*pointers, *self.args)
        return True


def fused_kv_compress_norm_rope_insert_sparse_attn_cutedsl(
    state_cache: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    cos_sin_cache: torch.Tensor,
    k_cache: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    kv_cache_block_size: int,
    kv_block_stride: int,
    head_size: int = 512,
    state_width: int = 1024,
    rope_head_dim: int = 64,
    fp8_max: float = 448.0,
    quant_block: int = 64,
    token_stride: int = 576,
    scale_dim: int = 8,
    compress_ratio: int = 4,
    overlap: bool = True,
    store_full_kv: bool = False,
    store_full_fp8: bool = False,
    fp8_scale: torch.Tensor | None = None,
) -> None:
    """Compress overlapping C4 windows and insert the boundary rows into k_cache.

    State rows hold two 512-element value banks followed by their logit banks.
    Packed pages have a separate scale plane. Full pages use the supplied
    tensor's byte strides. The upstream ABI retains kv_block_stride in both modes.
    """
    logger.debug("GEMS_NVIDIA FUSED_KV_COMPRESS_NORM_ROPE_INSERT_SPARSE_ATTN_CUTEDSL")
    global _last_cuda_plan
    if positions.numel() == 0:
        return
    if _guard_factory is not None:
        has_scale = store_full_fp8 and fp8_scale is not None
        tensors = (
            state_cache,
            token_to_req_indices,
            positions,
            slot_mapping,
            block_table,
            rms_norm_weight,
            cos_sin_cache,
            k_cache,
            kv_slot_mapping,
        )
        if has_scale:
            tensors += (fp8_scale,)
        options = (
            block_size,
            rms_norm_eps,
            type(rms_norm_eps),
            kv_cache_block_size,
            head_size,
            state_width,
            rope_head_dim,
            fp8_max,
            quant_block,
            token_stride,
            scale_dim,
            compress_ratio,
            overlap,
            store_full_kv,
            store_full_fp8,
            has_scale,
        )
        plan = _last_cuda_plan
        if plan is not None and plan.try_launch(tensors, options):
            return
    else:
        tensors = options = ()
    result = _dispatch_c4(
        state_cache,
        token_to_req_indices,
        positions,
        slot_mapping,
        block_table,
        block_size,
        rms_norm_weight,
        rms_norm_eps,
        cos_sin_cache,
        k_cache,
        kv_slot_mapping,
        kv_cache_block_size,
        head_size,
        state_width,
        rope_head_dim,
        fp8_max,
        quant_block,
        token_stride,
        scale_dim,
        compress_ratio,
        overlap,
        store_full_kv,
        store_full_fp8,
        fp8_scale,
        tuned_kernel=_fused_c4,
    )

    if (
        result is not None
        and tensors
        and torch.cuda.current_device() == state_cache.device.index
        and _hooks_clear()
        and all(type(t) is torch.Tensor for t in tensors)
        and reduce(or_, map(torch.Tensor.data_ptr, tensors)) % _fused_c4.divisibility
        == 0
    ):
        launch, grid, kernel_args = result
        compiled = launch[0]
        signature = getattr(getattr(compiled, "src", None), "signature", {})
        # Older Triton launchers differ in whether they accept constexprs.
        if (
            tuple(signature) == tuple(_fused_c4_kernel.arg_names)
            and signature["FP8_MAX"] == "constexpr"
        ):
            try:
                _last_cuda_plan = _CudaPlan(
                    tensors, options, compiled, grid, kernel_args
                )
            except (AttributeError, TypeError, RuntimeError):
                pass


__all__ = ["fused_kv_compress_norm_rope_insert_sparse_attn_cutedsl"]

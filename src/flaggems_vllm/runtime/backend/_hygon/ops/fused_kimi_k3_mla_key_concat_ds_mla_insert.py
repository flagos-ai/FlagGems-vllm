# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors

import logging
from functools import reduce
from operator import or_
from typing import NamedTuple, Optional

import torch
import triton

from flaggems_vllm.ops.fused_kimi_k3_mla_key_concat_ds_mla_insert import (
    _fixed_kernel,
    _prefill,
)

logger = logging.getLogger(__name__)

_plans = {}
_tensor_guards = None
_within_2gb = None
_current_device = getattr(torch._C, "_cuda_getDevice", None)
_current_stream = getattr(torch._C, "_cuda_getCurrentRawStream", None)
if (
    triton.__version__.split(".")[:2] == ["3", "6"]
    and torch.__version__.split(".")[:2] == ["2", "9"]
    and hasattr(triton.backends, "hcu")
):
    try:
        from triton import knobs
        from triton.backends.hcu.compiler import HIPBackend

        _hooks = knobs.runtime
        _buffer_ops = knobs.hcu
        _within_2gb = HIPBackend.is_within_2gb
        if (
            hasattr(_hooks.launch_enter_hook, "calls")
            and hasattr(_hooks.launch_exit_hook, "calls")
            and hasattr(_buffer_ops, "use_buffer_ops")
            and callable(_within_2gb)
            and callable(_current_device)
            and callable(_current_stream)
        ):
            _tensor_guards = torch._C._dynamo.guards.TensorGuards
    except (ImportError, AttributeError, TypeError):
        # The backend package also loads on HCU builds with different ABIs.
        pass


class _Launch(NamedTuple):
    check: object
    types: tuple
    device: int
    ranges: tuple
    run: object  # CompiledKernel's bound launcher owns its device module.
    scalars: tuple


def _hooks_active():
    return _hooks.launch_enter_hook.calls or _hooks.launch_exit_hook.calls


def _pointer_ranges(tensors):
    # FlagTree uses a different pointer specialization for HCU buffer ops
    # when the entire backing allocation fits in signed 32-bit offsets.
    return tuple(map(_within_2gb, tensors)) if _buffer_ops.use_buffer_ops else ()


def _select_config(q, head_tile):
    tokens, heads, _ = q.shape
    if (
        _tensor_guards
        and tokens * heads >= 4096
        and torch.cuda.get_device_name(q.device) == "BW"
    ):
        return (16 if heads % 16 == 0 else head_tile), {"num_warps": 2}
    return head_tile, {}


def _remember(compiled, programs, tensors, scalars, page, rope):
    if _tensor_guards is None:
        return
    try:
        if (
            type(compiled.run).__module__ != "triton.backends.hcu.driver"
            or type(compiled.run).__name__ != "HIPLauncher"
            or compiled.metadata.target.arch != "gfx936"
            or _hooks_active()
            or any(type(t) is not torch.Tensor for t in tensors)
            or reduce(or_, map(torch.Tensor.data_ptr, tensors))
            % _fixed_kernel.divisibility
        ):
            return
        check = _tensor_guards(
            *tensors, dynamic_dims_sizes=None, dynamic_dims_strides=None
        ).check
        if not check(*tensors):
            return
        plan = _Launch(
            check,
            tuple(map(type, tensors)),
            tensors[0].device.index,
            _pointer_ranges(tensors),
            compiled[programs, 1, 1],
            scalars,
        )
    except Exception:
        # The validated call already ran. Failure to construct an optional
        # launch plan must never replay its in-place writes.
        return
    if len(_plans) >= 64:
        _plans.clear()
    _plans[tensors[0].numel(), page, rope] = plan


def _launch(tensors, page, rope):
    plan = _plans.get((tensors[0].numel(), page, rope))
    try:
        if (
            plan is None
            or tuple(map(type, tensors)) != plan.types
            or not plan.check(*tensors)
            or _current_device() != plan.device
            or _hooks_active()
            or reduce(or_, map(torch.Tensor.data_ptr, tensors))
            % _fixed_kernel.divisibility
            or _pointer_ranges(tensors) != plan.ranges
        ):
            return False
        stream = _current_stream(plan.device)
    except Exception:
        # An unavailable ABI can fall back until the kernel has been enqueued.
        return False
    if not rope:
        tensors = (*tensors, tensors[6], tensors[2])
    plan.run(*tensors, *plan.scalars, stream=stream)
    return True


def fused_kimi_k3_mla_key_concat_ds_mla_insert(
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
    """Write full keys, optionally rotate Q in place, and insert 656-byte cache rows."""
    logger.debug("GEMS_HYGON FUSED_KIMI_K3_MLA_KEY_CONCAT_DS_MLA_INSERT")
    rope = position_ids is not None
    if rope != (cos_sin_cache is not None):
        raise ValueError("position_ids and cos_sin_cache must be provided together")
    tensors = (q, k_nope, k_pe, kv_c_normed, k_out, k_cache, slot_mapping)
    if rope:
        tensors += (position_ids, cos_sin_cache)
    if (
        _plans
        and type(q) is torch.Tensor
        and type(cache_block_size) is int
        and _launch(tensors, cache_block_size, rope)
    ):
        return
    _prefill(tensors, cache_block_size, rope, _remember, _select_config)

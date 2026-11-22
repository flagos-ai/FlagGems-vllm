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

"""Reuse compiled launchers for Ascend attention."""

from operator import itemgetter

import torch
import torch_npu
import triton
from triton.runtime import driver

_hook_chain_type = getattr(getattr(triton, "knobs", None), "HookChain", None)
_signatures = {}
_launchers = {}
_raw_stream_nowait = getattr(torch_npu._C, "_npu_getCurrentRawStreamNoWait", None)


_get_device_res_limit = getattr(torch_npu._C, "_npu_get_device_res_limit", None)


def cube_core_limit(device_index):
    if _get_device_res_limit is not None:
        # TorchNPU reads Cube limit as resource type zero and Vector as one.
        # Inputs already belong to an initialized NPU.
        return _get_device_res_limit(device_index, 0)
    return torch.npu.get_device_limit(device_index)["cube_core_num"]


def has_launch_hook(hook):
    # An empty FlagTree HookChain is callable but is always truthy.
    if type(hook) is _hook_chain_type:
        return bool(hook.calls)
    return bool(hook)


def bind_runner(entry, grid, device, direct=True):
    runner = entry[grid]
    runtime = getattr(getattr(triton, "knobs", None), "runtime", None)
    launch = getattr(entry, "run", None)
    function = getattr(entry, "function", None)
    metadata = getattr(entry, "packed_metadata", None)
    if (
        not direct
        or runtime is None
        or _raw_stream_nowait is None
        or not callable(launch)
        or function is None
        or not isinstance(metadata, dict)
    ):
        return lambda args: runner(
            *args, stream=driver.active.get_current_stream(device)
        )

    # Use the normal Triton wrapper when it must produce hook metadata.
    def submit(args):
        stream = _raw_stream_nowait(device)
        if has_launch_hook(runtime.launch_enter_hook) or has_launch_hook(
            runtime.launch_exit_hook
        ):
            runner(*args, stream=stream)
        else:
            launch(*grid, stream, function, metadata, None, None, None, *args)

    return submit


def submit_mixed(kernel, grid, device, direct, compact_key, *args, **kwargs):
    if (
        getattr(kernel, "pre_run_hooks", ())
        or getattr(kernel, "debug", False)
        or getattr(
            getattr(getattr(triton, "knobs", None), "runtime", None), "debug", False
        )
        or torch.npu.current_device() != device
    ):
        return kernel[grid](*args, **kwargs)

    signature_key = (kernel, compact_key)
    params, names, constants, dynamic = _signatures.get(signature_key, (None,) * 4)
    if params is None:
        params = tuple(
            (p.name, p.default) for p in kernel.signature.parameters.values()
        )
        names = frozenset(name for name, _ in params)
        constants, dynamic = None, ()
        if compact_key:
            triton_params = getattr(kernel, "params", ())
            static_indices = tuple(
                index for index, param in enumerate(triton_params) if param.is_constexpr
            )
            if len(static_indices) >= 2:
                constants = itemgetter(*static_indices)
                dynamic = tuple(
                    index for index in range(len(params)) if index not in static_indices
                )
        _signatures[signature_key] = params, names, constants, dynamic
    tail = params[len(args) :]
    compiler_options = tuple(
        (name, value) for name, value in kwargs.items() if name not in names
    )
    bound = args + tuple(kwargs.get(name, default) for name, default in tail)
    # Preserve data types, pointer alignment, and scalar specialization without
    # retaining actual tensor addresses or values.
    values = bound if constants is None else (bound[index] for index in dynamic)
    types_and_values = tuple(
        (
            (value.dtype, value.data_ptr() % 32)
            if isinstance(value, torch.Tensor)
            else value
        )
        for value in values
    )
    # For tight launch loops, collect compile-time scalars without going
    # through the dtype and alignment check used for live tensor arguments.
    static_values = constants(bound) if constants is not None else ()
    key = (
        kernel,
        grid,
        device,
        direct,
        compiler_options,
        static_values,
        types_and_values,
    )
    runner = _launchers.get(key)
    if runner is None:
        compiled = kernel[grid](*args, **kwargs)
        if len(_launchers) < 512 and hasattr(compiled, "__getitem__"):
            _launchers[key] = bind_runner(compiled, (grid + (1, 1))[:3], device, direct)
        return compiled
    runner(bound)

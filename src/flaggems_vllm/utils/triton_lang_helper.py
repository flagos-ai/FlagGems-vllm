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

from types import ModuleType

import triton
import triton.language as tl

from flaggems_vllm.runtime import backend
from flaggems_vllm.runtime.backend.device import DeviceDetector

"""
    To be compatible with different versions of math libraries
    tl_extra_shim will be selected to a specific library.
    And the "triton.language.extra" module is only available in
    Triton 2.2 and later versions.
"""

device = DeviceDetector()
backend.set_torch_backend_device_fn(device.vendor_name)
try:
    backend.set_tl_extra_backend_module(device.vendor_name)
    tl_extra_shim = backend.get_tl_extra_backend_module()
except ImportError:
    try:
        tl_extra_shim = triton.language.extra.libdevice
    except AttributeError:
        try:
            tl_extra_shim = triton.language.math
        except ImportError:
            tl_extra_shim = triton.language.libdevice


@triton.jit
def _fallback_pow(x, exponent: tl.constexpr):
    # Library callers use constant integer powers (squares and cubes). Core
    # Triton 3.6 has no tensor.__pow__, so do not rely on Python's ** lowering.
    tl.static_assert(isinstance(exponent, int), "expected a constant integer exponent")
    tl.static_assert(exponent >= 0, "expected a nonnegative exponent")
    result = tl.full(x.shape, 1, x.dtype)
    for _ in tl.static_range(exponent):
        result = result * x
    return result


@triton.jit
def _fallback_tanh(x):
    return 2.0 / (1.0 + tl.exp(-2.0 * x)) - 1.0


def _complete_math_shim(module):
    # Match FlagGems' pure Triton fallbacks without modifying vendor modules.
    fallbacks = {"pow": _fallback_pow, "tanh": _fallback_tanh}
    missing = {name: fn for name, fn in fallbacks.items() if not hasattr(module, name)}
    if not missing:
        return module
    shim = ModuleType(module.__name__)
    shim.__dict__.update(vars(module))
    shim.__dict__.update(missing)
    return shim


tl_extra_shim = _complete_math_shim(tl_extra_shim)


def use_backend(module):
    """using backend module impl"""

    def decorator(func):
        func_name = func.__name__
        if hasattr(module, func_name):
            try:
                return getattr(module, func_name)
            except Exception:
                pass
        return func

    return decorator


def use_tl_extra(func):
    """backend function shim"""
    return use_backend(tl_extra_shim)(func)

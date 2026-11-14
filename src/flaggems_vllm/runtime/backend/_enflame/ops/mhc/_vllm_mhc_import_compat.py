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

"""Import guard for vLLM's mHC layer module on non-CUDA platforms.

vLLM 0.20.0 gates ``tilelang`` behind ``current_platform.is_cuda_alike()`` in
``vllm/model_executor/layers/mhc.py``: on every other platform it assigns
``tilelang = None`` and then applies ``@tilelang.jit(...)`` at module level,
so merely importing the module raises
``AttributeError: 'NoneType' object has no attribute 'jit'``. Installing
tilelang does not help (the gate is the platform check, which is always False
on GCU), and the FlagGems tests/benchmarks that probe the module do so with
``except ImportError``, which cannot catch AttributeError. As a result pytest
collection of ``tests/test_mhc_ops.py``, ``tests/test_mhc_pre_npu.py`` and
``benchmark/test_mhc.py`` aborts on every GCU host (upstream vLLM bug; the
correct upstream behavior would be raising ImportError or guarding the
decorator).

This module installs a meta-path finder that, for exactly the
``vllm.model_executor.layers.mhc`` import, lets the real module load and only
translates that known NoneType-jit AttributeError into a chained ImportError,
which the existing ``except ImportError`` probes then handle as "vLLM
reference unavailable". Any other outcome passes through unchanged: the
module imports normally wherever tilelang is usable, other AttributeErrors
propagate untouched, no other module is intercepted, and installing the
guard never imports vLLM itself.

Remove this shim once vLLM raises ImportError on non-CUDA platforms (or the
tests' probes widen to AttributeError).
"""

import sys
from importlib.machinery import PathFinder

_TARGET_MODULE = "vllm.model_executor.layers.mhc"


class _GuardedLoader:
    """Delegate to the real loader; translate the known crash to ImportError."""

    def __init__(self, real_loader):
        self._real_loader = real_loader

    def create_module(self, spec):
        return self._real_loader.create_module(spec)

    def exec_module(self, module):
        try:
            self._real_loader.exec_module(module)
        except AttributeError as exc:
            if "NoneType" not in str(exc) or "'jit'" not in str(exc):
                raise
            raise ImportError(
                "vllm.model_executor.layers.mhc is not importable on this "
                "host: vLLM sets tilelang=None on non-CUDA platforms and "
                "then applies @tilelang.jit at module level (upstream bug); "
                "original error: {}".format(exc)
            ) from exc


class _VllmMhcImportGuardFinder:
    """Meta-path finder active for exactly ``vllm.model_executor.layers.mhc``."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname != _TARGET_MODULE or _TARGET_MODULE in sys.modules:
            return None
        spec = PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _GuardedLoader(spec.loader)
        return spec


def install_vllm_mhc_import_guard():
    """Install the vLLM mHC import guard finder (idempotent)."""
    for finder in sys.meta_path:
        if isinstance(finder, _VllmMhcImportGuardFinder):
            return
    sys.meta_path.insert(0, _VllmMhcImportGuardFinder())

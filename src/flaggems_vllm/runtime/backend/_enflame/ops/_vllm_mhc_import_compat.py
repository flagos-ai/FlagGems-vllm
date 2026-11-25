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

"""Import compatibility for vLLM's mHC layer module on the enflame GCU backend.

``tests/test_mhc_ops.py`` (plus ``tests/test_mhc_pre_npu.py`` and
``benchmark/test_mhc.py``) import ``vllm.model_executor.layers.mhc`` inside
``try: ... except ImportError:`` blocks so the optional vLLM comparison path
degrades to a skip when the module is unavailable.  vLLM's own module cannot
satisfy that contract on this host: ``vllm/model_executor/layers/mhc.py``
gates ``tilelang`` on ``current_platform.is_cuda_alike()``, pins
``tilelang = None`` on every other platform, and then unconditionally applies
``@tilelang.jit(...)`` at module level, so on the GCU the import dies with
``AttributeError: 'NoneType' object has no attribute 'jit'`` — an exception
the ``except ImportError`` blocks cannot catch, which aborts pytest collection
outright (baseline COLLECTION_ERROR).

Installing tilelang does not help (the gate is the platform check, which is
False for the GCU regardless), so this is an upstream vLLM bug, not a missing
dependency.  This module installs a meta-path finder that, for exactly the
``vllm.model_executor.layers.mhc`` import, lets the real module load and
converts only the known ``'NoneType' object has no attribute 'jit'`` failure
into the ImportError vLLM itself raises on the cuda-alike branch when tilelang
is missing, restoring the graceful degradation the tests were written for.  On
hosts where the module imports natively (CUDA) the finder is a transparent
passthrough.

The parallel mhc_pre_npu task landed an equivalent guard inside
``ops/mhc/_vllm_mhc_import_compat.py`` (7ae821a); this copy keeps the
``ops/mhc`` test gate self-contained on worktrees based before that commit
(dedup candidate once the tree settles).  Remove this shim once vLLM raises
ImportError (or guards the tilelang.jit decoration) on non-CUDA platforms.
"""

import sys

_VLLM_MHC_MODULE = "vllm.model_executor.layers.mhc"
# vLLM's non-CUDA-alike branch pins tilelang to None and then decorates with
# @tilelang.jit; that exact attribute error is the only signature we convert.
_TILELANG_NONE_MARKER = "'NoneType' object has no attribute 'jit'"


class _MhcImportCompatLoader:
    """Load the real module; convert the known tilelang-gate crash."""

    def __init__(self, real_loader):
        self._real_loader = real_loader

    def create_module(self, spec):
        return self._real_loader.create_module(spec)

    def exec_module(self, module):
        try:
            self._real_loader.exec_module(module)
        except AttributeError as exc:
            if _TILELANG_NONE_MARKER not in str(exc):
                raise
            # Mirror vLLM's own "tilelang is required" ImportError so callers
            # guarding with ``except ImportError`` degrade gracefully instead
            # of losing the whole collection.
            raise ImportError(
                "vllm.model_executor.layers.mhc is unavailable on this "
                "platform: vLLM pins tilelang to None on non-CUDA-alike "
                "platforms and then applies @tilelang.jit at module level "
                f"({exc})"
            ) from exc


class _VllmMhcImportCompatFinder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _VLLM_MHC_MODULE or fullname in sys.modules:
            return None
        try:
            start = sys.meta_path.index(self) + 1
        except ValueError:
            return None
        for finder in sys.meta_path[start:]:
            spec = finder.find_spec(fullname, path)
            if spec is not None and spec.loader is not None:
                # Wrap the loader in place: rebuilding a ModuleSpec would drop
                # has_location and leave the module without __file__.
                spec.loader = _MhcImportCompatLoader(spec.loader)
                return spec
        return None


def install_vllm_mhc_import_compat():
    """Install the vLLM mHC import compatibility finder (idempotent)."""
    for finder in sys.meta_path:
        if isinstance(finder, _VllmMhcImportCompatFinder):
            return
    sys.meta_path.insert(0, _VllmMhcImportCompatFinder())

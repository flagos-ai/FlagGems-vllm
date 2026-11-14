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

"""Import compatibility for vLLM's vendored DeepGEMM extension.

The GCU image ships a CPU-only torch build, so the vendored
``vllm.third_party.deep_gemm._C`` CUDA extension cannot be dlopen'ed (its
NEEDED libraries ``libnvrtc.so.13`` / ``libc10_cuda.so`` / ``libtorch_cuda.so``
do not exist here). vLLM still gates every DeepGEMM call path off on this
host (``is_deep_gemm_supported()`` is False), but a few FlagGems test and
benchmark modules import ``vllm.third_party.deep_gemm.utils.math`` (pure-torch
helpers) at module level; that transitively executes ``deep_gemm/__init__.py``
and aborts pytest collection with ImportError.

This module installs a meta-path finder that, for exactly the
``vllm.third_party.deep_gemm._C`` import, first lets the real extension load.
Only when that fails with a missing-shared-object error does it fall back to
a stub module providing just the names that ``deep_gemm/__init__.py`` and
``utils/layout.py`` import unconditionally; the optional kernel imports keep
raising ImportError and are swallowed by vLLM's own try blocks, so a stubbed
host reports "DeepGEMM unavailable" instead of pretending to work. On hosts
where the real extension loads, the finder is a transparent passthrough.

Remove this shim once the image ships a CUDA-capable torch.
"""

import os
import sys
from importlib.machinery import ModuleSpec

_STUB_MODULE_NAME = "vllm.third_party.deep_gemm._C"
_MISSING_SO_MARKER = "cannot open shared object file"

# Names imported from _C outside any try block (deep_gemm/__init__.py and
# utils/layout.py). Everything else is optional and must stay absent so the
# "optional kernels" try blocks keep reporting them as unavailable.
_UNCONDITIONAL_NAMES = (
    "set_num_sms",
    "get_num_sms",
    "set_tc_util",
    "get_tc_util",
    "set_ignore_compile_dims",
    "set_block_size_multiple_of",
    "set_pdl",
    "get_pdl",
    "cublaslt_gemm_nt",
    "cublaslt_gemm_nn",
    "cublaslt_gemm_tn",
    "cublaslt_gemm_tt",
    "set_mk_alignment_for_contiguous_layout",
    "get_mk_alignment_for_contiguous_layout",
    "get_theoretical_mk_alignment_for_contiguous_layout",
)


def _deep_gemm_unavailable(*args, **kwargs):
    raise NotImplementedError(
        "vllm.third_party.deep_gemm._C is a flaggems_vllm import stub: the "
        "real CUDA extension could not be loaded on this host (missing "
        "shared libraries), so no DeepGEMM kernel is available"
    )


def _fallback_cuda_home():
    try:
        import torch

        candidate = os.path.normpath(
            os.path.join(os.path.dirname(torch.__file__), "..", "nvidia", "cu13")
        )
        if os.path.isdir(candidate):
            return candidate
    except ImportError:
        pass
    return "/usr"


def _populate_stub(module):
    for name in _UNCONDITIONAL_NAMES:
        setattr(module, name, _deep_gemm_unavailable)
    module.init = lambda *args, **kwargs: None
    module.__flaggems_deep_gemm_stub__ = True
    # deep_gemm/__init__.py ends with _C.init(lib_root, _find_cuda_home());
    # _find_cuda_home() asserts unless CUDA_HOME/CUDA_PATH, nvcc or
    # /usr/local/cuda exists, none of which do on a CUDA-less host.
    os.environ.setdefault("CUDA_HOME", _fallback_cuda_home())


class _StubLoader:
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        _populate_stub(module)


class _FallbackLoader:
    """Load the real extension; fall back to the stub on missing libraries."""

    def __init__(self, real_loader):
        self._real_loader = real_loader
        self._stubbed = False

    def create_module(self, spec):
        try:
            return self._real_loader.create_module(spec)
        except ImportError as exc:
            if _MISSING_SO_MARKER not in str(exc):
                raise
            self._stubbed = True
            return None

    def exec_module(self, module):
        if self._stubbed:
            _populate_stub(module)
        else:
            self._real_loader.exec_module(module)


class _DeepGemmCompatFinder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _STUB_MODULE_NAME or _STUB_MODULE_NAME in sys.modules:
            return None
        try:
            start = sys.meta_path.index(self) + 1
        except ValueError:
            return None
        for finder in sys.meta_path[start:]:
            spec = finder.find_spec(fullname, path)
            if spec is not None and spec.loader is not None:
                return ModuleSpec(
                    fullname, _FallbackLoader(spec.loader), origin=spec.origin
                )
        return ModuleSpec(fullname, _StubLoader())


def install_vllm_deep_gemm_import_compat():
    """Install the DeepGEMM import compatibility finder (idempotent)."""
    if _STUB_MODULE_NAME in sys.modules:
        return
    for finder in sys.meta_path:
        if isinstance(finder, _DeepGemmCompatFinder):
            return
    sys.meta_path.insert(0, _DeepGemmCompatFinder())

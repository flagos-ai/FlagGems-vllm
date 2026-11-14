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

"""Runtime compatibility for vLLM's paged MQA logits wrappers on GCU.

The GCU deep_gemm build hard-exits the whole process when its C++ entry
points reject an argument, which kills the pytest session before the
operator under test is even called:

1. ``deep_gemm_cpp.get_paged_mqa_logits_metadata`` asserts
   ``num_sms == <backend SM count>`` (24 on this SKU). vLLM's accuracy
   tests query ``torch.cuda.get_device_properties(0).multi_processor_count``
   (torch_gcu reports the SPC count, e.g. 4) and pass it straight through,
   so every test dies at the metadata call with
   "Error: csrc/apis/attention_gcu.hpp : get_paged_mqa_logits_metadata,
   argument num_sms must be equal to 24" and no pytest summary is produced.

2. ``fp8_fp4_paged_mqa_logits`` (topsaten C++ path) rejects the standard
   vLLM 2D ``[B*next_n, H]`` weights / ``[B, max_blocks]`` block-table
   ranks with "rank of output and weights and block_table must be equal"
   followed by another hard exit. The package ships a torch-native
   implementation for exactly this layout, selected per-call by the
   vendor-documented ``ENFLAME_DEEPGEMM_KERNEL_TRITON_BACKEND`` switch.

This module installs a lazy, import-time-only compat layer:

* a meta-path post-import hook on ``vllm.utils.deep_gemm`` that wraps
  ``get_paged_mqa_logits_metadata`` to substitute the caller-supplied SM
  count with the deep_gemm backend's own reported value when they differ
  (the metadata only schedules the CUDA-side kernel; the GCU math path
  validates its shape for parity and does not consume it);
* the ``ENFLAME_DEEPGEMM_KERNEL_TRITON_BACKEND`` entry for
  ``fp8_fp4_paged_mqa_logits`` so the reference resolves to the
  torch-native implementation.

Both pieces are no-ops on hosts where the external deep_gemm package is
not importable (the calls then proceed exactly as before). Remove this
shim once torch_gcu stops hard-exiting on mismatched num_sms and accepts
the standard vLLM 2D layout in the topsaten path.
"""

import os
import sys

_TARGET_MODULE = "vllm.utils.deep_gemm"

_TRITON_BACKEND_ENV = "ENFLAME_DEEPGEMM_KERNEL_TRITON_BACKEND"
_PAGED_KERNEL_NAME = "fp8_fp4_paged_mqa_logits"


def _backend_num_sms():
    """SM count reported by the deep_gemm backend itself (None if absent)."""
    try:
        import deep_gemm

        return int(deep_gemm.get_num_sms())
    except Exception:
        return None


def _patch_vllm_deep_gemm_module(module):
    """Wrap ``get_paged_mqa_logits_metadata`` to clamp num_sms per backend."""
    original = module.get_paged_mqa_logits_metadata
    if getattr(original, "_flaggems_paged_mqa_compat", False):
        return

    def get_paged_mqa_logits_metadata(context_lens, block_size, num_sms):
        backend_num_sms = _backend_num_sms()
        if backend_num_sms is not None and int(num_sms) != backend_num_sms:
            num_sms = backend_num_sms
        return original(context_lens, block_size, num_sms)

    get_paged_mqa_logits_metadata._flaggems_paged_mqa_compat = True
    module.get_paged_mqa_logits_metadata = get_paged_mqa_logits_metadata


class _PostImportLoader:
    """Delegate to the real loader, then patch the freshly-executed module."""

    def __init__(self, real_loader):
        self._real_loader = real_loader

    def create_module(self, spec):
        return self._real_loader.create_module(spec)

    def exec_module(self, module):
        self._real_loader.exec_module(module)
        _patch_vllm_deep_gemm_module(module)


class _VllmDeepGemmPagedCompatFinder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _TARGET_MODULE or _TARGET_MODULE in sys.modules:
            return None
        try:
            start = sys.meta_path.index(self) + 1
        except ValueError:
            return None
        for finder in sys.meta_path[start:]:
            spec = finder.find_spec(fullname, path)
            if spec is not None and spec.loader is not None:
                spec.loader = _PostImportLoader(spec.loader)
                return spec
        return None


def _enable_torch_backend_for_paged_mqa_logits():
    """Route the paged MQA logits reference to the torch-native path."""
    current = os.environ.get(_TRITON_BACKEND_ENV, "")
    names = {name.strip() for name in current.replace(",", " ").split()}
    if _PAGED_KERNEL_NAME in names:
        return
    names.add(_PAGED_KERNEL_NAME)
    os.environ[_TRITON_BACKEND_ENV] = " ".join(sorted(names))


def install_vllm_paged_mqa_logits_compat():
    """Install the paged MQA logits GCU compat layer (idempotent)."""
    _enable_torch_backend_for_paged_mqa_logits()
    module = sys.modules.get(_TARGET_MODULE)
    if module is not None:
        _patch_vllm_deep_gemm_module(module)
        return
    for finder in sys.meta_path:
        if isinstance(finder, _VllmDeepGemmPagedCompatFinder):
            return
    sys.meta_path.insert(0, _VllmDeepGemmPagedCompatFinder())

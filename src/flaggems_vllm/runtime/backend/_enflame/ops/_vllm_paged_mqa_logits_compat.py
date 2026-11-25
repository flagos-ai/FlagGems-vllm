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
* int32-safe replacements for the FP4 codecs in
  ``vllm.third_party.deep_gemm.utils.math`` when the int64 env is enabled
  (``torch.bucketize`` and the arange-derived group index hard-abort GCU
  processes there; see ``_patch_vllm_deep_gemm_math_module``).

Both pieces are no-ops on hosts where the external deep_gemm package is
not importable (the calls then proceed exactly as before). Remove this
shim once torch_gcu stops hard-exiting on mismatched num_sms and accepts
the standard vLLM 2D layout in the topsaten path.
"""

import os
import sys

import torch

_TARGET_MODULE = "vllm.utils.deep_gemm"
_MATH_MODULE = "vllm.third_party.deep_gemm.utils.math"

_TRITON_BACKEND_ENV = "ENFLAME_DEEPGEMM_KERNEL_TRITON_BACKEND"
_PAGED_KERNEL_NAME = "fp8_fp4_paged_mqa_logits"
_INT64_ENV = "TORCH_GCU_ENABLE_INT64_AND_UINT64"


def _int64_env_enabled() -> bool:
    """Whether torch_gcu keeps Long tensors as real int64 in this process."""
    value = os.environ.get(_INT64_ENV, "")
    return value.strip().lower() in ("1", "true", "on", "yes")


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


def _patch_vllm_deep_gemm_math_module(module):
    """Replace the int64-hazard FP4 codecs with int32-safe equivalents.

    With ``TORCH_GCU_ENABLE_INT64_AND_UINT64=1`` the default-dtype int64
    tensors inside the vendor codecs route to unimplemented topsaten kernels
    and hard-abort the process (no Python exception):

    * ``torch.bucketize`` (int64 output) in ``_quantize_to_fp4_e2m1`` — the
      abort that kills the FP4 accuracy cases before the op under test runs
      (P21 instance vllm_fp8_fp4_paged_mqa_logits_bucketize);
    * the ``torch.arange``-derived int64 group index in ``cast_back_from_fp4``.

    The replacements keep the math identical: counting the strictly lower
    boundaries in int32 is exactly ``bucketize(..., right=False)`` for a
    strictly increasing boundary list, and the group index stays within
    int32 for any realistic head dimension.
    """
    if not _int64_env_enabled():
        return
    if getattr(module.cast_back_from_fp4, "_flaggems_gcu_int32_safe", False):
        return

    def _quantize_to_fp4_e2m1(x):
        ax = x.abs().clamp_max(6.0)
        boundaries = torch.tensor(
            [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=x.device, dtype=ax.dtype
        )
        idx = (ax.unsqueeze(-1) > boundaries).sum(dim=-1, dtype=torch.int32)
        code = idx.to(torch.uint8)
        sign = (x < 0) & (idx != 0)
        code = code | (sign.to(torch.uint8) << 3)
        return code.view(torch.int8)

    _unpack_ue8m0_from_int = module.unpack_ue8m0_from_int
    _dequantize_from_fp4_e2m1 = module._dequantize_from_fp4_e2m1

    def cast_back_from_fp4(
        packed, sf, gran_k: int = 128, use_packed_ue8m0: bool = False
    ):
        m, n2 = packed.shape
        n = n2 * 2
        if use_packed_ue8m0:
            sf = _unpack_ue8m0_from_int(sf)
        unpacked = torch.zeros((m, n), dtype=torch.int8, device=packed.device)
        unpacked[:, ::2] = packed & 0x0F
        unpacked[:, 1::2] = (packed >> 4) & 0x0F
        x_dequantized = _dequantize_from_fp4_e2m1(unpacked)
        group_idx = torch.arange(n, device=packed.device, dtype=torch.int32) // gran_k
        return x_dequantized * sf[:, group_idx]

    _quantize_to_fp4_e2m1._flaggems_gcu_int32_safe = True
    cast_back_from_fp4._flaggems_gcu_int32_safe = True
    module._quantize_to_fp4_e2m1 = _quantize_to_fp4_e2m1
    module.cast_back_from_fp4 = cast_back_from_fp4


class _PostImportLoader:
    """Delegate to the real loader, then patch the freshly-executed module."""

    def __init__(self, real_loader, patch_fn):
        self._real_loader = real_loader
        self._patch_fn = patch_fn

    def create_module(self, spec):
        return self._real_loader.create_module(spec)

    def exec_module(self, module):
        self._real_loader.exec_module(module)
        self._patch_fn(module)


_TARGETS = {
    _TARGET_MODULE: _patch_vllm_deep_gemm_module,
    _MATH_MODULE: _patch_vllm_deep_gemm_math_module,
}


class _VllmDeepGemmPagedCompatFinder:
    def find_spec(self, fullname, path=None, target=None):
        patch_fn = _TARGETS.get(fullname)
        if patch_fn is None or fullname in sys.modules:
            return None
        try:
            start = sys.meta_path.index(self) + 1
        except ValueError:
            return None
        for finder in sys.meta_path[start:]:
            spec = finder.find_spec(fullname, path)
            if spec is not None and spec.loader is not None:
                spec.loader = _PostImportLoader(spec.loader, patch_fn)
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
    math_module = sys.modules.get(_MATH_MODULE)
    if math_module is not None:
        _patch_vllm_deep_gemm_math_module(math_module)
    module = sys.modules.get(_TARGET_MODULE)
    if module is not None:
        _patch_vllm_deep_gemm_module(module)
        return
    for finder in sys.meta_path:
        if isinstance(finder, _VllmDeepGemmPagedCompatFinder):
            return
    sys.meta_path.insert(0, _VllmDeepGemmPagedCompatFinder())

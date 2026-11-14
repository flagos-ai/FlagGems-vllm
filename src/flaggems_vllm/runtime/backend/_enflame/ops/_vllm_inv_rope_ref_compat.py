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

"""vLLM fused_inv_rope_fp8_quant reference compatibility for the enflame GCU
backend.

``benchmark/test_fused_inv_rope_fp8_quant.py`` times vLLM's own
``fused_inv_rope_fp8_quant`` as the baseline leg.  Its kernel loads
``positions`` (int64) with a scalar ``tl.load`` whose result feeds address
arithmetic, which aborts the device on GCU (compiler issue
``triton_i64_narrow_load``); verified end-to-end on L600 / vllm 0.20.0 /
torch_gcu 2.11.0 that the identical op runs cleanly once positions are
int32.  The wrapper installed here re-exposes int64 positions as the compact
int32 storage words (zero-copy view, no kernel launch so the timed baseline
closure is not skewed) before delegating to vLLM's implementation -- exactly
what the FlagGems host does for its own kernel.  Positions are provably
< 2**31, so the values are bit-identical; the baseline kernel, grid and
launch parameters stay untouched.

The wrapper is a no-op passthrough on non-GCU devices and when
``TORCH_GCU_ENABLE_INT64_AND_UINT64=1`` (true i64 storage; narrow i64 loads
work in that mode).  Remove this shim once vLLM loads positions in a
GCU-safe width or torch_gcu fixes narrow i64 scalar loads.
"""

import sys

_VLLM_INV_ROPE_MODULES = (
    "vllm.v1.attention.ops.deepseek_v4_ops",
    "vllm.v1.attention.ops.deepseek_v4_ops.fused_inv_rope_fp8_quant",
    "vllm.models.deepseek_v4.common.ops",
    "vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant",
)


def _wrap_fused_inv_rope_fp8_quant(orig):
    from flaggems_vllm.runtime.backend._enflame.ops.fused_inv_rope_fp8_quant import (
        _positions_as_int32,
    )

    def fused_inv_rope_fp8_quant(o, positions, *args, **kwargs):
        positions = _positions_as_int32(positions)
        return orig(o, positions, *args, **kwargs)

    fused_inv_rope_fp8_quant.__wrapped__ = orig
    fused_inv_rope_fp8_quant.__doc__ = orig.__doc__
    # Keep the vLLM origin: benchmark/_validate_baseline refuses baselines
    # whose module resolves into flaggems_vllm.
    fused_inv_rope_fp8_quant.__module__ = orig.__module__
    fused_inv_rope_fp8_quant._flaggems_gcu_inv_rope_ref_compat = True
    return fused_inv_rope_fp8_quant


def _patch_module(module):
    fn = getattr(module, "fused_inv_rope_fp8_quant", None)
    if fn is None or not callable(fn):
        return
    if getattr(fn, "_flaggems_gcu_inv_rope_ref_compat", False):
        return
    if getattr(fn, "__module__", "").startswith("flaggems"):
        return
    module.fused_inv_rope_fp8_quant = _wrap_fused_inv_rope_fp8_quant(fn)


class _PatchingLoader:
    def __init__(self, real_loader):
        self._real_loader = real_loader

    def create_module(self, spec):
        return self._real_loader.create_module(spec)

    def exec_module(self, module):
        self._real_loader.exec_module(module)
        try:
            _patch_module(module)
        except Exception:
            pass


class _VllmInvRopeRefCompatFinder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _VLLM_INV_ROPE_MODULES or fullname in sys.modules:
            return None
        try:
            start = sys.meta_path.index(self) + 1
        except ValueError:
            return None
        for finder in sys.meta_path[start:]:
            spec = finder.find_spec(fullname, path)
            if spec is not None and spec.loader is not None:
                # Wrap the loader in place: rebuilding a ModuleSpec would drop
                # has_location and leave the module without __file__ (same
                # caveat as _vllm_moe_ref_compat).
                spec.loader = _PatchingLoader(spec.loader)
                return spec
        return None


def install_vllm_inv_rope_ref_compat():
    """Install the vLLM fused_inv_rope_fp8_quant GCU compat (idempotent)."""
    for name in _VLLM_INV_ROPE_MODULES:
        module = sys.modules.get(name)
        if module is not None:
            _patch_module(module)
    for finder in sys.meta_path:
        if isinstance(finder, _VllmInvRopeRefCompatFinder):
            return
    sys.meta_path.insert(0, _VllmInvRopeRefCompatFinder())

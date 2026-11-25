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

"""``LibEntry.cache_key`` compatibility for the enflame GCU backend.

Upstream tests (``tests/test_tle_capabilities.py::test_attn_res_missing_tle_
load_uses_non_tle_cache_key``) assert that every kernel decorated with
``@libentry()`` exposes a truthy ``cache_key``, mirroring the contract
``LibTuner`` already implements in ``flaggems_vllm.utils.libentry``: the
``cache_key`` of the innermost ``triton.runtime.JITFunction`` (a hash of the
kernel source and its global dependencies).  The ``LibEntry`` wrapper class
synced from FlagGems does not define that attribute, so any
``@libentry()``-wrapped kernel (e.g. ``attn_res._attn_res_post_kernel``)
raises ``AttributeError`` on attribute access.

This is a plain flaggems_vllm Python-level API gap, not a triton_gcu
runtime/compiler problem: ``JITFunction.cache_key`` exists and works on
FlagTree/triton_gcu 3.6.0, and the missing attribute is backend-agnostic.
The patch below is additive and idempotent: it only defines the property
when ``LibEntry`` does not already provide it, and it delegates to
``self.jit_function`` — the innermost JITFunction that ``LibEntry.__init__``
already resolves — so the value matches ``LibTuner.cache_key`` semantics
exactly.  Remove this shim once the synced ``libentry.py`` carries
``cache_key`` on ``LibEntry``.
"""


def install_flaggems_libentry_compat():
    """Expose ``cache_key`` on ``LibEntry`` (idempotent, additive only)."""
    from flaggems_vllm.utils.libentry import LibEntry

    if hasattr(LibEntry, "cache_key"):
        return

    def _cache_key(self):
        return self.jit_function.cache_key

    LibEntry.cache_key = property(_cache_key)

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

"""Extension operators for FlagGems-vllm.

This package hosts fused operators that are explicitly opt-in: nothing here is
part of ``flaggems_vllm._FULL_CONFIG``, so importing it never patches PyTorch
dispatch. Callers import an operator directly::

    from flaggems_vllm.extensions import my_fused_op

Layout::

    extensions/
      __init__.py         # public entry points (one per operator)
      _dispatch.py        # backend resolution: arch -> vendor -> generic
      _registry.py        # @register_extension / @register_backend_impl
      ops/<op>.py         # generic ("公版") implementation, one file per op
      backends/_<vendor>[/<arch>]/ops/<op>.py   # optional overrides

Writing an operator:

1. Implement the kernel and host dispatch in ``ops/<op>.py`` and decorate the
   public function with ``@register_extension("<op>")``.
2. Export it from ``ops/__init__.py`` (the package ``__all__``) and add a
   dispatching entry point in this module's ``__all__``.
3. Optional: add ``backends/_<vendor>/ops/<op>.py`` using
   ``@register_backend_impl("<op>", vendor="<vendor>")`` to specialize one
   backend, or ``backends/_<vendor>/<arch>/ops/<op>.py`` for one architecture.
4. Add ``tests/test_extensions/test_<op>.py`` with ``@pytest.mark.<op>`` and
   ``benchmark/test_extensions/test_<op>.py``, plus a ``conf/operators.yaml``
   entry with the ``extension`` label.

Rules that still apply (see ``workflow.md``): no torch compute fallback in
production paths, NVIDIA autotune configs are mandatory for kernels with
performance parameters, and unsupported paths raise ``NotImplementedError``.
"""

from flaggems_vllm.extensions._dispatch import current_arch, current_vendor
from flaggems_vllm.extensions._registry import (
    EXTENSION_REGISTRY,
    ExtensionEntry,
    list_extensions,
    make_dispatcher,
    register_backend_impl,
    register_extension,
    resolve,
)

__all__ = [
    "EXTENSION_REGISTRY",
    "ExtensionEntry",
    "current_arch",
    "current_vendor",
    "list_extensions",
    "make_dispatcher",
    "register_backend_impl",
    "register_extension",
    "resolve",
]

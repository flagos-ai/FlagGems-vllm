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

"""Multi-backend resolution for extension operators.

Extension operators are implemented in
``flaggems_vllm/extensions/ops/<op>.py`` as the generic ("公版") Triton
implementation and may be specialized per vendor or per vendor architecture
under ``flaggems_vllm/extensions/backends/_<vendor>[/<arch>]/ops/<op>.py``.

Resolution order, first hit wins::

    backends/_<vendor>/<arch>/ops   # arch specialization, e.g. _nvidia/hopper
    backends/_<vendor>/ops          # vendor specialization, e.g. _nvidia
    ops                             # generic implementation

Vendor packages are optional: a vendor without specializations simply has no
``backends/_<vendor>/`` directory and is skipped. Adding a new backend never
requires editing this module.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import Iterator, Optional, Tuple

from flaggems_vllm import runtime

EXTENSION_ROOT_PKG = "flaggems_vllm.extensions"
EXTENSION_BACKENDS_PKG = f"{EXTENSION_ROOT_PKG}.backends"
GENERIC_OPS_MODULE = f"{EXTENSION_ROOT_PKG}.ops"

# Modules already imported (or confirmed absent) by this process.
_visited: set = set()


def current_vendor() -> str:
    """Return the vendor name of the active backend, e.g. ``"nvidia"``."""
    return runtime.device.vendor_name


def current_arch() -> Optional[str]:
    """Return the architecture name of the active device, e.g. ``"hopper"``.

    Returns ``None`` when the backend does not publish an ``ARCH_MAP`` or the
    device cannot be queried (CPU-only hosts, accelerators without arch
    specializations). Arch detection must never break package import, hence the
    broad catch.
    """
    try:
        arch_event = runtime.backend.BackendArchEvent()
    except Exception:  # noqa: BLE001
        return None
    if not getattr(arch_event, "has_arch", False):
        return None
    return getattr(arch_event, "arch", None)


def vendor_ops_module(vendor: str, arch: Optional[str] = None) -> str:
    """Return the module name of the vendor/arch ops package for *vendor*."""
    vendor_pkg = vendor if vendor.startswith("_") else f"_{vendor}"
    parts = [EXTENSION_BACKENDS_PKG, vendor_pkg]
    if arch:
        parts.append(arch)
    parts.append("ops")
    return ".".join(parts)


def import_ops_module(module_name: str) -> bool:
    """Import *module_name* once.

    Returns ``True`` when the module exists and is imported, ``False`` when it
    does not exist. A module that exists but fails to import raises, so a real
    error inside a kernel is never silently swallowed.
    """
    if module_name in _visited:
        return True
    try:
        spec = importlib.util.find_spec(module_name)
    except ModuleNotFoundError:
        spec = None
    if spec is None:
        _visited.add(module_name)
        return False
    importlib.import_module(module_name)
    _visited.add(module_name)
    return True


def dispatch_order(
    vendor: Optional[str] = None, arch: Optional[str] = None
) -> Iterator[Tuple[Optional[str], Optional[str]]]:
    """Yield ``(vendor, arch)`` pairs from most to least specialized.

    ``None`` for ``vendor`` means "generic". ``arch=None`` with a vendor set
    means "vendor-wide specialization, no arch directory".
    """
    if vendor:
        if arch:
            yield vendor, arch
        yield vendor, None
    yield None, None


def load_impls(vendor: Optional[str] = None, arch: Optional[str] = None) -> None:
    """Import the ops modules that may register implementations for a backend.

    The generic aggregator is always imported first so that per-file
    registrations exist before vendor modules (which usually override a
    generic name) are executed.
    """
    import_ops_module(GENERIC_OPS_MODULE)
    if vendor:
        if arch:
            import_ops_module(vendor_ops_module(vendor, arch))
        import_ops_module(vendor_ops_module(vendor))


__all__ = [
    "EXTENSION_BACKENDS_PKG",
    "EXTENSION_ROOT_PKG",
    "GENERIC_OPS_MODULE",
    "current_arch",
    "current_vendor",
    "dispatch_order",
    "import_ops_module",
    "load_impls",
    "vendor_ops_module",
]

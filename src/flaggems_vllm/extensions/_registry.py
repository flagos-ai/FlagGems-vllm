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

"""Registry for extension operators and their per-backend implementations.

A single operator is registered once per implementation layer:

* ``register_extension`` — the generic (公版) implementation, runnable on the
  upstream Triton compiler of every supported backend.
* ``register_backend_impl`` — an optional vendor or vendor/arch override that
  shadows the generic one for that backend only.

Lookups go through :func:`resolve`, which honours
:func:`flaggems_vllm.extensions._dispatch.dispatch_order`. An operator without
any implementation for the current device raises ``NotImplementedError``: there
is deliberately no torch compute fallback (see ``workflow.md`` §2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

from flaggems_vllm.extensions import _dispatch

ImplKey = Tuple[str, str]

#: Key of the generic implementation in :attr:`ExtensionEntry.impls`.
GENERIC_KEY: ImplKey = ("", "")


def _impl_key(vendor: Optional[str], arch: Optional[str]) -> ImplKey:
    return (vendor or "", arch or "")


def _label(key: ImplKey) -> str:
    vendor, arch = key
    if not vendor:
        return "generic"
    return f"{vendor}/{arch}" if arch else vendor


@dataclass
class ExtensionEntry:
    """All implementations registered for one extension operator."""

    name: str
    #: Names of the ATen / vLLM entry points this operator implements, if any.
    #: Library-only operators (no dispatch override) leave this empty.
    aten_ops: Tuple[str, ...] = ()
    impls: Dict[ImplKey, Callable] = field(default_factory=dict)
    _cache: Dict[ImplKey, Optional[Callable]] = field(default_factory=dict)

    def register(
        self,
        impl: Callable,
        vendor: Optional[str] = None,
        arch: Optional[str] = None,
    ) -> Callable:
        if arch and not vendor:
            raise ValueError(
                f"extension operator '{self.name}': an arch override "
                f"({arch!r}) also requires its vendor"
            )
        if not callable(impl):
            raise TypeError(
                f"extension operator '{self.name}': implementation must be "
                f"callable, got {type(impl).__name__}"
            )
        key = _impl_key(vendor, arch)
        existing = self.impls.get(key)
        if existing is not None and existing is not impl:
            raise ValueError(
                f"extension operator '{self.name}': duplicate implementation "
                f"for backend {_label(key)}"
            )
        self.impls[key] = impl
        self._cache.clear()
        return impl

    def backend_labels(self) -> Tuple[str, ...]:
        return tuple(sorted(_label(key) for key in self.impls)) or ("<none>",)

    def resolve(
        self, vendor: Optional[str] = None, arch: Optional[str] = None
    ) -> Callable:
        """Return the best implementation for a backend (current device by default)."""
        if vendor is None:
            vendor, arch = _dispatch.current_vendor(), _dispatch.current_arch()
        key = _impl_key(vendor, arch)
        if key not in self._cache:
            self._load_backend_overrides(vendor, arch)
            self._cache[key] = self._lookup(vendor, arch)
        impl = self._cache[key]
        if impl is None:
            raise NotImplementedError(
                f"extension operator '{self.name}' is not implemented for "
                f"backend {_label(key)}; registered implementations: "
                f"{', '.join(self.backend_labels())}"
            )
        return impl

    def _load_backend_overrides(
        self, vendor: Optional[str], arch: Optional[str]
    ) -> None:
        """Import the backend modules that may register overrides, before lookup.

        Registration happens as an import side effect, so a vendor or arch
        module that has not been imported yet cannot be found by ``_lookup``.
        This must run before every uncached lookup rather than once at import
        time: importing the whole backend tree eagerly would pull in kernels
        for hardware that is not present.
        """
        _dispatch.load_impls(vendor, arch)

    def _lookup(self, vendor: Optional[str], arch: Optional[str]) -> Optional[Callable]:
        for candidate_vendor, candidate_arch in _dispatch.dispatch_order(vendor, arch):
            impl = self.impls.get(_impl_key(candidate_vendor, candidate_arch))
            if impl is not None:
                return impl
        return None


#: Published so callers can introspect which extension operators exist.
EXTENSION_REGISTRY: Dict[str, ExtensionEntry] = {}


def _get_entry(name: str, aten_ops: Tuple[str, ...] = ()) -> ExtensionEntry:
    entry = EXTENSION_REGISTRY.get(name)
    if entry is None:
        entry = ExtensionEntry(name=name, aten_ops=tuple(aten_ops))
        EXTENSION_REGISTRY[name] = entry
    elif aten_ops:
        entry.aten_ops = tuple(aten_ops)
    return entry


def register_extension(name: str, impl: Optional[Callable] = None, *, aten_ops=()):
    """Register the generic implementation of an extension operator.

    Usable as a decorator::

        @register_extension("my_fused_op")
        def my_fused_op(...): ...

    or directly as ``register_extension("my_fused_op", my_fused_op)``.
    """

    def _decorator(fn: Callable) -> Callable:
        _get_entry(name, aten_ops).register(fn)
        return fn

    return _decorator if impl is None else _decorator(impl)


def register_backend_impl(
    name: str, vendor: str, arch: Optional[str] = None
) -> Callable:
    """Decorator registering a vendor (or vendor/arch) override for *name*."""
    if not vendor:
        raise ValueError(
            f"extension operator '{name}': register_backend_impl needs a vendor; "
            f"use register_extension for the generic implementation"
        )

    def _decorator(fn: Callable) -> Callable:
        _get_entry(name).register(fn, vendor=vendor, arch=arch)
        return fn

    return _decorator


def resolve(
    name: str, vendor: Optional[str] = None, arch: Optional[str] = None
) -> Callable:
    """Return the implementation of extension operator *name* for a backend.

    With no ``vendor``, the current device's vendor and arch are used. Passing
    ``vendor`` explicitly disables arch detection, so ``arch=None`` there means
    "vendor-wide implementation".
    """
    entry = EXTENSION_REGISTRY.get(name)
    if entry is None:
        # The operator may simply not be imported yet.
        _dispatch.load_impls(vendor, arch)
        entry = EXTENSION_REGISTRY.get(name)
    if entry is None:
        raise NotImplementedError(
            f"unknown extension operator '{name}'; known operators: "
            f"{', '.join(sorted(EXTENSION_REGISTRY)) or '<none>'}"
        )
    return entry.resolve(vendor=vendor, arch=arch)


def make_dispatcher(name: str) -> Callable:
    """Build a thin entry point that dispatches to *name*'s backend impl."""

    def _dispatch_entry(*args, **kwargs):
        return resolve(name)(*args, **kwargs)

    _dispatch_entry.__name__ = name
    _dispatch_entry.__qualname__ = name
    _dispatch_entry.__doc__ = (
        f"Backend-dispatching entry point for extension operator '{name}'."
    )
    _dispatch_entry.__flaggems_extension__ = name
    return _dispatch_entry


def list_extensions() -> Tuple[str, ...]:
    """Return the sorted names of the registered extension operators."""
    _dispatch.import_ops_module(_dispatch.GENERIC_OPS_MODULE)
    return tuple(sorted(EXTENSION_REGISTRY))


__all__ = [
    "EXTENSION_REGISTRY",
    "ExtensionEntry",
    "list_extensions",
    "make_dispatcher",
    "register_backend_impl",
    "register_extension",
    "resolve",
]

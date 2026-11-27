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

"""Compile-option keying of the Ascend eager launcher cache.

The launchers in `sparse_attn_sharedkv.py` cache a compiled kernel's runner and
call it without re-entering LibEntry, so LibEntry's own option check never runs
on a hit. These tests drive the real launcher functions against a real
`LibEntry`, with a fake in place of the compile, so they need neither an NPU nor
a GPU.
"""

import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import triton
from triton import language as tl

import flaggems_vllm
from flaggems_vllm.utils import libentry

_MODULE_PATH = (
    Path(flaggems_vllm.__file__).parent
    / "runtime/backend/_ascend/ops/sparse_attn_sharedkv.py"
)

# Packages that only a CANN / Ascend FlagTree build provides.
_CANN_ONLY = (
    "triton.extension",
    "triton.extension.buffer",
    "triton.extension.buffer.language",
    "triton.language.extra.cann",
    "triton.language.extra.cann.extension",
    "triton.language.extra.cann.libdevice",
)


class _Stub(types.ModuleType):
    """Stand-in namespace: public attributes resolve to further stubs."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        child = _Stub(f"{self.__name__}.{name}")
        setattr(self, name, child)
        return child

    def __call__(self, *args, **kwargs):
        return self


def _stub(name):
    module = _Stub(name)
    module.__path__ = []
    module.__spec__ = importlib.machinery.ModuleSpec(name, None)
    return module


def _importable(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


@pytest.fixture(scope="module")
def sas():
    """Load the Ascend module by path, stubbing only what this host lacks.

    Loading by path skips `_ascend/ops/__init__`, which imports every Ascend op.
    """
    with pytest.MonkeyPatch.context() as patch:
        for name in _CANN_ONLY:
            if not _importable(name):
                patch.setitem(sys.modules, name, _stub(name))
        if not _importable("triton.runtime.libentry"):
            passthrough = types.ModuleType("triton.runtime.libentry")
            passthrough.libentry = lambda *args, **kwargs: lambda fn: fn
            patch.setitem(sys.modules, "triton.runtime.libentry", passthrough)
        name = "sparse_attn_sharedkv_under_test"
        spec = importlib.util.spec_from_file_location(name, _MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        patch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        yield module


@pytest.fixture(autouse=True)
def isolated_launcher_cache(sas, monkeypatch):
    monkeypatch.setattr(sas, "_EAGER_COMPILED_LAUNCHER_CACHE", {})
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(current_device=lambda: 0), raising=False
    )


OPTIONS_A = {"multibuffer": False, "unit_flag": True}
OPTIONS_B = {"multibuffer": True, "unit_flag": True}
GRID = (1,)
DISPATCH_KEY = ("dtype", "aligned")


@triton.jit
def _kernel(n, BLOCK: tl.constexpr):
    pass


class _Compiled:
    """A compiled kernel remembering the options it was built with."""

    def __init__(self, options, launches):
        self.options = options
        self.launches = launches

    def __getitem__(self, grid3):
        return lambda *args, **kwargs: self.launches.append(("cached", self.options))


@pytest.fixture
def launches():
    return []


@pytest.fixture
def entry(launches, monkeypatch):
    """A real `LibEntry` whose compile step is faked.

    The fake stores what it builds the way `LibEntry.run` does: under the
    argument key from `LibEntry.key` plus `LibEntry._launch_options_key`.
    """
    entry = libentry()(_kernel)
    entry.kernel_cache = ({},)

    def run(*args, grid, warmup, **kwargs):
        options = {k: v for k, v in kwargs.items() if k not in entry._kernel_params}
        launches.append(("entry", options))
        key = entry.key([args[0]], [], [kwargs["BLOCK"]])
        key += entry._launch_options_key(kwargs)
        entry.kernel_cache[0].setdefault(key, (_Compiled(options, launches),))

    monkeypatch.setattr(entry, "run", run)
    return entry


@pytest.mark.sparse_attn_sharedkv
def test_compile_options_key_is_order_independent_and_value_sensitive(sas):
    key = sas._compile_options_key
    assert key(OPTIONS_A) == key(dict(reversed(OPTIONS_A.items())))
    assert key(OPTIONS_A) != key(OPTIONS_B)
    assert key(OPTIONS_A) != key({**OPTIONS_A, "optimize_epilogue": False})
    assert key({}) == ()
    hash(key({"cluster_dims": [1, 1, 1]}))


@pytest.mark.sparse_attn_sharedkv
def test_launcher_cache_key_separates_compile_options(sas, entry):
    def key(options, grid=GRID):
        return sas._eager_launcher_cache_key(0, entry, grid, options, DISPATCH_KEY)

    assert key(OPTIONS_A) == key(dict(reversed(OPTIONS_A.items())))
    assert key(OPTIONS_A) != key(OPTIONS_B)
    assert key(OPTIONS_A) != key(OPTIONS_A, grid=(2,))


@pytest.mark.sparse_attn_sharedkv
def test_cached_launcher_does_not_serve_other_compile_options(sas, entry, launches):
    """The same kernel, grid and dispatch key built twice, with different options."""
    for options in (OPTIONS_A, OPTIONS_A, OPTIONS_B, OPTIONS_B, OPTIONS_A):
        sas._launch_cached_eager_kernel(
            entry, GRID, (8,), ("BLOCK",), (64,), options, DISPATCH_KEY
        )

    assert launches == [
        ("entry", OPTIONS_A),
        ("cached", OPTIONS_A),
        ("entry", OPTIONS_B),
        ("cached", OPTIONS_B),
        ("cached", OPTIONS_A),
    ]


@pytest.mark.sparse_attn_sharedkv
def test_q256_launcher_does_not_serve_other_compile_options(sas, entry, launches):
    for options in (OPTIONS_A, OPTIONS_A, OPTIONS_B, OPTIONS_B):
        sas._launch_q256_cached_eager_kernel(
            (0, 1234), entry, GRID, (8,), ("BLOCK",), (64,), options, DISPATCH_KEY
        )

    assert launches == [
        ("entry", OPTIONS_A),
        ("cached", OPTIONS_A),
        ("entry", OPTIONS_B),
        ("cached", OPTIONS_B),
    ]


@pytest.mark.sparse_attn_sharedkv
@pytest.mark.parametrize("stream_key", [None, (0, 1234)])
def test_scfa_pipeline_launcher_does_not_serve_other_compile_options(
    sas, entry, launches, monkeypatch, stream_key
):
    monkeypatch.setattr(
        sas, "_sparse_attn_sharedkv_prefill_scfa_static_slot_pipeline_kernel", entry
    )
    monkeypatch.setattr(sas, "_SCFA_STATIC_SLOT_TRAILING_NAMES", ("BLOCK",))

    for options in (OPTIONS_A, OPTIONS_A, OPTIONS_B, OPTIONS_B):
        sas._launch_q256_cached_scfa_score_guard_pipeline(
            stream_key, GRID, (8,), (64,), options, DISPATCH_KEY
        )

    # With a stream the q256 launcher serves the hit; without one the generic
    # launcher does. Both key on the compile options.
    assert launches == [
        ("entry", OPTIONS_A),
        ("cached", OPTIONS_A),
        ("entry", OPTIONS_B),
        ("cached", OPTIONS_B),
    ]


@pytest.mark.sparse_attn_sharedkv
def test_entry_without_launch_options_key_keeps_the_cached_launcher(sas, launches):
    """FlagTree's own `triton.runtime.libentry` keys on arguments alone.

    The kernels in this module use that one. Asking it for launch options would
    raise, and the launcher would then drop to the slow path for good.
    """

    class ArgumentOnlyEntry:
        specialize_indices = [0]
        do_not_specialize_indices = []
        kernel_cache = ({},)

        def key(self, spec_args, dns_args, const_args):
            return tuple(spec_args + dns_args + const_args)

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches.append(("entry", None))
                key = self.key([args[0]], [], [kwargs["BLOCK"]])
                self.kernel_cache[0].setdefault(key, (_Compiled(None, launches),))

            return launch

    # One instance: the launcher cache is keyed on `id(entry)`.
    entry = ArgumentOnlyEntry()
    for _ in range(3):
        sas._launch_cached_eager_kernel(
            entry, GRID, (8,), ("BLOCK",), (64,), OPTIONS_A, DISPATCH_KEY
        )

    assert launches == [("entry", None), ("cached", None), ("cached", None)]

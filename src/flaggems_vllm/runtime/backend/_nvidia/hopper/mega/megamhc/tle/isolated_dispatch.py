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

"""Setup-only loader for independently specialized MegaMHC token buckets.

Some kernel modules read their exact geometry from the environment at import
time.  Context construction serializes that mutable setup, creates a direct
compiled launcher, and restores process state.  The returned launcher owns
the compiled code and never consults the module registry or environment on
the operator hot path.
"""

from __future__ import annotations

import os
import sys
import threading
from contextlib import contextmanager
from types import ModuleType
from typing import Iterator, Mapping

from .plan_launch_spec import launch_spec_for_plan
from .prepared_launcher import PreparedCompiledLauncher, prepare_launch_spec

_TLE_PACKAGE = __package__
_KERNEL_PREFIX = f"{_TLE_PACKAGE}.kernels."
_MUTABLE_TOP_LEVEL_MODULES = {f"{_TLE_PACKAGE}.fused_post_gemm_t64"}
_ENV_PREFIX = "DG_TLE_MHC_"
_INITIALIZATION_LOCK = threading.Lock()


@contextmanager
def _clean_tle_environment() -> Iterator[None]:
    saved = {
        name: value
        for name, value in os.environ.items()
        if name.startswith(_ENV_PREFIX)
    }
    for name in saved:
        os.environ.pop(name, None)
    try:
        yield
    finally:
        for name in tuple(os.environ):
            if name.startswith(_ENV_PREFIX):
                os.environ.pop(name, None)
        os.environ.update(saved)


def _unlink_parent_attribute(name: str, module: ModuleType) -> None:
    parent_name, separator, child_name = name.rpartition(".")
    if not separator:
        return
    parent = sys.modules.get(parent_name)
    if parent is not None and getattr(parent, child_name, None) is module:
        delattr(parent, child_name)


def _evict_variant_modules() -> None:
    names = sorted(
        (
            name
            for name in sys.modules
            if name.startswith(_KERNEL_PREFIX)
            or name in _MUTABLE_TOP_LEVEL_MODULES
        ),
        key=lambda name: name.count("."),
        reverse=True,
    )
    for name in names:
        module = sys.modules.pop(name, None)
        if isinstance(module, ModuleType):
            _unlink_parent_attribute(name, module)


def prepare_isolated_safe_launcher(
    inputs: Mapping[str, object],
    outputs: Mapping[str, object],
) -> PreparedCompiledLauncher:
    """Compile one exact-T launcher while isolating import-time variants."""
    from . import dispatch

    tokens = dispatch._tokens_from_inputs(inputs)
    with _INITIALIZATION_LOCK, _clean_tle_environment():
        _evict_variant_modules()
        dispatch._loaded_case = None
        try:
            prepare = dispatch._load_prepare(tokens)
            plan = prepare(inputs, outputs)
            return prepare_launch_spec(launch_spec_for_plan(plan))
        finally:
            dispatch._loaded_case = None


__all__ = ["prepare_isolated_safe_launcher"]
